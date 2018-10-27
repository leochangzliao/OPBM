# --------------------------------------------------------
# Faster R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick and Sean Bell
# --------------------------------------------------------

import caffe
import numpy as np
import yaml
from fast_rcnn.config import cfg
from generate_anchors import generate_anchors
from fast_rcnn.bbox_transform import bbox_transform_inv, clip_boxes
from fast_rcnn.nms_wrapper import nms
from utils.cython_bbox_grid import bbox_overlaps_grid
from utils.cython_bbox import bbox_overlaps
import os
import cPickle
import logging
import matplotlib.pyplot as plt

cache_path = './data/cache/'
logging.basicConfig(filename=cache_path + 'msg.log')
DEBUG = False


class PseudoGtLayer(caffe.Layer):
    """
    Outputs object detection proposals by applying estimated bounding-box
    transformations to a set of regular boxes (called "anchors").
    """
    iter_times = 0

    def setup(self, bottom, top):
        # parse the layer parameter string, which must be valid YAML
        layer_params = yaml.load(self.param_str_)
        self.object_number = layer_params['object_number']  # object_number =20
        self.rpn_pse_gt_list_average = []
        self.rpn_pse_gt_list_feature = []
        if DEBUG:
            print 'object_number: {:}'.format(self.object_number)
            # rois blob: holds R regions of interest, each is a 5-tuple
            # (n, x1, y1, x2, y2) specifying an image batch index n and a
            # rectangle (x1, y1, x2, y2)
            # top[0].reshape(1, 3)

    def forward(self, bottom, top):
        rpn_rois = bottom[0].data
        simu_points = bottom[1].data
        flipped = bottom[2].data
        img_index = bottom[3].data
        img_info = bottom[4].data[0]
        img_data = bottom[5].data
        gt_boxes = bottom[6].data
        rpn_scores = bottom[7].data
        hyper_rpn_conv_features = bottom[8].data
        if DEBUG:
            print '+++++++++++++++++++++++++++++++++++++++++++++++++++++'
            print 'rpn_rois: ({}, {},{})'.format(len(rpn_rois), rpn_rois[0], np.shape(rpn_rois))
            print 'rpn_scores:{},{}'.format(len(rpn_scores), rpn_scores[0], np.shape(rpn_scores))
            print 'simu_points: {},{}'.format(len(simu_points), simu_points)
            print 'flipped:{}'.format(flipped)
            print 'img_index:{}'.format(img_index)
            print 'img_info:{}'.format(img_info)
            print 'gt_boxes:{}'.format(gt_boxes)
            print 'hyper_rpn_conv_features:{}'.format(np.shape(hyper_rpn_conv_features))
            print '+++++++++++++++++++++++++++++++++++++++++++++++++++++'
        rpn_boxes = rpn_rois[:, 1:5]
        one_epoc = 10022
        rpn_simu_gt_boxes_average, rpn_simu_gt_boxes_feature_map = generate_rpn_simu_gt_boxes(img_data, simu_points,
                                                                                              gt_boxes, img_info,
                                                                                              rpn_boxes, rpn_scores,
                                                                                              flipped,
                                                                                              hyper_rpn_conv_features,
                                                                                              )
        # print 'rpn_simu_gt_boxes:{}'.format(rpn_simu_gt_boxes)
        if len(self.rpn_pse_gt_list_average) < one_epoc:
            self.rpn_pse_gt_list_average.append({'img_index': int(img_index),
                                                 'rpn_simu_gt_boxes': rpn_simu_gt_boxes_average})
            self.rpn_pse_gt_list_feature.append({'img_index': int(img_index),
                                                 'rpn_simu_gt_boxes': rpn_simu_gt_boxes_feature_map})
        if len(self.rpn_pse_gt_list_average) == one_epoc:
            if self.iter_times + 1 == one_epoc:
                self.rpn_pse_gt_list_average.sort(key=take_key)
                self.rpn_pse_gt_list_feature.sort(key=take_key)
                # cache_file = os.path.join(cache_path +
                #                           'voc_2007_trainval_rpn_pse_gt_boxes_hyper_merged_1_' + str(
                #     self.iter_times + 1) + '.pkl')
                # with open(cache_file, 'wb') as fid:
                #     cPickle.dump(self.rpn_pse_gt_list_average, fid, cPickle.HIGHEST_PROTOCOL)
            elif self.iter_times + 1 > one_epoc:
                save_pse_results(self, self.rpn_pse_gt_list_average, img_index, rpn_simu_gt_boxes_average, one_epoc,
                                 'average_')
                save_pse_results(self, self.rpn_pse_gt_list_feature, img_index, rpn_simu_gt_boxes_feature_map, one_epoc,
                                 'hyper_merged_')
        self.iter_times += 1
        # print self.iter_times, "___________________"

    def backward(self, top, propagate_down, bottom):
        """This layer does not propagate gradients."""
        pass

    def reshape(self, bottom, top):
        """Reshaping happens during the call to forward."""
        pass


def take_key(elem):
    return elem['img_index']


def save_pse_results(self, pse_gt_list, img_index, rpn_simu_gt_boxes, one_epoc, name):
    order = [rpn_boxes['img_index'] for rpn_boxes in self.rpn_pse_gt_list_average]
    i_dx = order.index(int(img_index))
    temp1 = pse_gt_list[i_dx]
    temp2 = pse_gt_list[i_dx + 1]
    if (temp1['rpn_simu_gt_boxes'][:, 4] == rpn_simu_gt_boxes[0, 4]).all():
        pse_gt_list[i_dx] = {'img_index': int(img_index),
                             'rpn_simu_gt_boxes': rpn_simu_gt_boxes}
    elif (temp2['rpn_simu_gt_boxes'][:, 4] == rpn_simu_gt_boxes[0, 4]).all():
        pse_gt_list[i_dx + 1] = {'img_index': int(img_index),
                                 'rpn_simu_gt_boxes': rpn_simu_gt_boxes}
    if self.iter_times % one_epoc == 0:
        cache_file = os.path.join(cache_path +
                                  'voc_2007_trainval_rpn_pse_gt_boxes_' + str(name) + str(
            self.iter_times + 1) + '.pkl')
        with open(cache_file, 'wb') as fid:
            cPickle.dump(pse_gt_list, fid, cPickle.HIGHEST_PROTOCOL)


def filter_rpn_boxes(rpn_boxes, rpn_scores, simu_point, img_info, gt_box, hyper_rpn_conv_features):
    Top_N_Boxes_small = 5
    Top_N_Boxes_big = 10
    area_ratio_small = 0.1
    area_ratio_big_low = 0.1
    area_ratio_big_high = 0.85
    # print 'rpn_boxes_num_original:{},{}'.format(len(rpn_boxes), len(rpn_scores))
    overlaps = bbox_overlaps(rpn_boxes.astype(np.float),
                             np.array(gt_box[0:4]).reshape((1, 4)).astype(np.float))
    keep = np.where(np.array(overlaps) >= 0.8)[0]
    rpn_boxes = rpn_boxes[keep, :]
    rpn_scores = rpn_scores[keep, :]

    # print 'rpn_boxes_num_after bbox_overlaps filter:{},{}'.format(len(rpn_boxes), len(rpn_scores))
    # box_xs1 = rpn_boxes[:, 0]
    # box_ys1 = rpn_boxes[:, 1]
    # box_xs2 = rpn_boxes[:, 2]
    # box_ys2 = rpn_boxes[:, 3]
    # object_area = (box_xs2 - box_xs1) * (box_ys2 - box_ys1) / (img_info[0] * img_info[1])
    # if int(simu_point[2]) == 1:  # small object
    #     keep = np.where((box_xs1 < simu_point[0]) & (box_xs2 > simu_point[0]) &
    #                     (box_ys1 < simu_point[1]) & (box_ys2 > simu_point[1]) & (object_area <= area_ratio_small))[0]
    # else:
    #     keep = np.where((box_xs1 < simu_point[0]) & (box_xs2 > simu_point[0]) &
    #                     (box_ys1 < simu_point[1]) & (box_ys2 > simu_point[1]) &
    #                     (object_area <= area_ratio_big_high) & (object_area >= area_ratio_big_low))[0]
    # if len(keep) == 0:
    #     margin = 5 * img_info[2]
    #     if int(simu_point[2]) == 1:  # small object
    #
    #         keep = np.where((box_xs1 - margin < simu_point[0]) & (box_xs2 + margin > simu_point[0]) &
    #                         (box_ys1 - margin < simu_point[1]) & (box_ys2 + margin > simu_point[1]) & (
    #                             object_area <= area_ratio_small))[0]
    #     else:
    #         keep = np.where((box_xs1 - margin < simu_point[0]) & (box_xs2 + margin > simu_point[0]) &
    #                         (box_ys1 - margin < simu_point[1]) & (box_ys2 + margin > simu_point[1]) &
    #                         (object_area <= area_ratio_big_high) & (object_area >= area_ratio_big_low))[0]
    # rpn_boxes = rpn_boxes[keep, :]
    # rpn_scores = rpn_scores[keep]
    # print 'rpn_boxes_num_after filter:{},{}'.format(len(rpn_boxes), len(rpn_scores))

    # 4. sort all (proposal, score) pairs by score from highest to lowest
    # 5. take top pre_nms_topN (e.g. 6000)
    order = rpn_scores.ravel().argsort()[::-1]
    pre_nms_topN = 2000
    if pre_nms_topN > 0:
        order = order[:pre_nms_topN]
    rpn_boxes = rpn_boxes[order, :]
    rpn_scores = rpn_scores[order]
    # print 'rpn_boxes_num_after rpn_scores:{}'.format(len(rpn_boxes))
    # do nms for big or small object
    keep = nms(np.hstack((rpn_boxes, rpn_scores)), 0.7)
    proposals = rpn_boxes[keep, :]
    rpn_scores = rpn_scores[keep]
    print 'filtered num after nms {}'.format(len(proposals))

    # box_xs1 = proposals[:, 0]
    # box_ys1 = proposals[:, 1]
    # box_xs2 = proposals[:, 2]
    # box_ys2 = proposals[:, 3]
    # rpn_boxes_ratio = np.max([[box_xs2 - simu_point[0] + 1], [simu_point[0] - box_xs1 + 1]], axis=0) / \
    #                   np.min([[box_xs2 - simu_point[0] + 1], [simu_point[0] - box_xs1 + 1]], axis=0) * \
    #                   np.max([[box_ys2 - simu_point[1] + 1], [simu_point[1] - box_ys1 + 1]], axis=0) / \
    #                   np.min([[box_ys2 - simu_point[1] + 1], [simu_point[1] - box_ys1 + 1]], axis=0)
    #
    # order = rpn_boxes_ratio.ravel().argsort()
    # if int(simu_point[2]) == 1:
    #     order = order[:Top_N_Boxes_small] if len(order) >= Top_N_Boxes_small else order
    # else:
    #     order = order[:Top_N_Boxes_big] if len(order) >= Top_N_Boxes_big else order
    # rpn_boxes_ratio = rpn_boxes_ratio.ravel()[order]
    # print rpn_boxes_ratio[:10]
    # proposals = proposals[order, :]
    # rpn_scores = rpn_scores[order]

    # print 'rpn_scores',rpn_scores,proposals
    if len(proposals) == 0:
        proposals = np.array(gt_box[0:4]).reshape((1, 4))
        merged_proposal = grid_feature_soft_max(hyper_rpn_conv_features, proposals, rpn_scores, int(simu_point[2]))
        return proposals, merged_proposal
    else:
        merged_proposal = grid_feature_soft_max(hyper_rpn_conv_features, proposals, rpn_scores, int(simu_point[2]))
        return proposals, merged_proposal


def grid_feature_soft_max(hyper_features, proposals, rpn_scores, is_small_object):
    feature_shape = np.shape(hyper_features)
    hyper_features = hyper_features.reshape((feature_shape[1], feature_shape[2], feature_shape[3]))
    pred_proposals = np.zeros((len(proposals), 4), np.float)
    for index, proposal in enumerate(proposals):
        proposal = proposal / 4  # feature map size
        height = int(proposal[3] - proposal[1]) + 1
        width = int(proposal[2] - proposal[0]) + 1
        grids = get_grids(height, width, int(proposal[1]), int(proposal[0]), feature_shape, is_small_object)
        # fig2, ax2 = plt.subplots(figsize=(12, 12))
        # gci = ax2.imshow(np.average(hyper_features, axis=0))
        # plt.colorbar(gci, fraction=0.046, pad=0.04)
        # ax2.add_patch(
        #     plt.Rectangle((proposal[0], proposal[1]),
        #                   proposal[2] - proposal[0],
        #                   proposal[3] - proposal[1], fill=False,
        #                   edgecolor='red', linewidth=0.5)
        # )
        # (chanel,height,width)
        # print np.shape(hyper_features),
        grid_value = [np.sum(hyper_features[:, grid[1]:grid[3], grid[0]:grid[2]]) for grid in grids]
        max_ = np.max(grid_value)
        # sum_ = np.sum(np.exp(grid_value - max_))
        # grid_value = np.exp(grid_value - max_) / sum_
        sum_ = np.sum(grid_value)
        grid_value = grid_value / sum_
        average_value = np.average(grid_value)
        # order = np.array(grid_value).ravel().argsort()[::-1]
        order = np.where(grid_value >= average_value)[0]
        # if is_small_object == 1:
        #     order = order[:20]
        # else:
        #     order = order[:40]
        grids = np.array(grids)[order, :]
        # for grid in grids:
        #     ax2.add_patch(
        #         plt.Rectangle((grid[0], grid[1]),
        #                       grid[2] - grid[0],
        #                       grid[3] - grid[1], fill=False,
        #                       edgecolor='yellow', linewidth=0.5)
        #     )
        pred_proposals[index] = [np.min(grids[:, 0]), np.min(grids[:, 1]), np.max(grids[:, 2]), np.max(grids[:, 3])]
        # bbox = pred_proposals[index]
        # ax2.add_patch(
        #
        #     plt.Rectangle((bbox[0], bbox[1]),
        #                   bbox[2] - bbox[0],
        #                   bbox[3] - bbox[1], fill=False,
        #                   edgecolor='white', linewidth=2)
        # )
        # plt.title('hyper_rpn_conv_features')
        # plt.show()
    # print np.mean(pred_proposals, axis=0),np.shape(np.mean(pred_proposals, axis=0))
    return np.mean(pred_proposals, axis=0).reshape((1, 4)) * 4


def generate_rpn_simu_gt_boxes(img_data, simu_points, gt_boxes, img_info, rpn_boxes, rpn_scores, flipped,
                               hyper_rpn_conv_features):
    rpn_simu_gt_boxes_average = np.zeros((len(simu_points), 5))
    rpn_simu_gt_boxes_feature_map = np.zeros((len(simu_points), 5))
    view_detect = True
    for i_gt in xrange(len(rpn_simu_gt_boxes_average)):
        simu_point = simu_points[i_gt]
        gt_box = gt_boxes[i_gt]
        rpn_boxes_filtered, merged_proposal = filter_rpn_boxes(rpn_boxes, rpn_scores, simu_point, img_info, gt_box,
                                                               hyper_rpn_conv_features)
        assert len(rpn_boxes_filtered) is not 0

        if view_detect:
            averge_boxes = np.average(rpn_boxes_filtered, axis=0)  # / img_info[2]
        else:
            averge_boxes = np.average(rpn_boxes_filtered, axis=0) / img_info[2]
            rpn_simu_gt_boxes_average[i_gt, :] = [averge_boxes[0], averge_boxes[1], averge_boxes[2], averge_boxes[3],
                                                  int(flipped)]
            merged = merged_proposal[0] / img_info[2]
            rpn_simu_gt_boxes_feature_map[i_gt, :] = [merged[0], merged[1], merged[2], merged[3], int(flipped)]

        # print '____________',averge_boxes
        '''
        im_grids = get_grid_bbox(int(img_info[0]), int(img_info[1]))
        grid_overlaps = bbox_overlaps_grid(im_grids.astype(np.float), np.array(rpn_boxes_filtered).astype(np.float))

        sum_grids = np.sum(grid_overlaps, axis=1)
        # print(im_grids.shape, grid_overlaps.shape, sum_grids.shape)
        _max = np.max(sum_grids)
        _min = np.min(sum_grids)
        pred_flag_init = False
        if int(simu_point[2]) is 1:
            alpha_thresh = 0.7
            # plt.title('small_object')
        else:
            alpha_thresh = 0.3
            # plt.title('large_object')
        # print(im.shape,gt_bbox,1.0*(gt_bbox[2]-gt_bbox[0])*(gt_bbox[3]-gt_bbox[1])/(im.shape[0]*im.shape[1]))
        for i_grid in range(len(im_grids)):
            grid = im_grids[i_grid]
            distance = get_distance(simu_point[0], simu_point[1], 0, 0, 0.5 *
                                    (grid[0] + grid[2]), 0.5 * (grid[1] + grid[3]))
            if int(simu_point[2]) is not 1:
                alpha = (sum_grids[i_grid] - _min) / (_max - _min) * np.exp(-1.0 * distance / (800 * img_info[2]))
            else:
                alpha = (sum_grids[i_grid] - _min) / (_max - _min) * np.cos(distance / (160 * img_info[2]))
                # np.exp(-1.0 * distance / 100)

            if alpha >= alpha_thresh:
                # ax.add_patch(
                #     plt.Rectangle((grid[0], grid[1]),
                #                   grid[2] - grid[0],
                #                   grid[3] - grid[1], fill=True,
                #                   color='yellow', alpha=alpha, edgecolor=None)
                # )
                if not pred_flag_init:
                    pred_xmin, pred_xmax, pred_ymin, pred_ymax = grid[0], grid[2], grid[1], grid[3]
                    pred_flag_init = True
                else:
                    if pred_xmin > grid[0]:
                        pred_xmin = grid[0]
                    if pred_ymin > grid[1]:
                        pred_ymin = grid[1]
                    if pred_xmax < grid[2]:
                        pred_xmax = grid[2]
                    if pred_ymax < grid[3]:
                        pred_ymax = grid[3]
        if pred_xmax >= img_info[1]:
            pred_xmax = pred_xmax - 1
        if pred_ymax >= img_info[0]:
            pred_ymax = pred_ymax - 1
        # if pred_xmin < 0 or pred_ymin < 0:
        #     print('pred_xmin:', pred_xmin, 'pred_ymin:', pred_ymin)
        rpn_simu_gt_boxes[i_gt, :5] = [pred_xmin , pred_ymin ,
                                       pred_xmax , pred_ymax ,
                                       int(flipped)]
        ax.add_patch(
            plt.Rectangle((rpn_simu_gt_boxes[i_gt, 0], rpn_simu_gt_boxes[i_gt, 1]),
                          rpn_simu_gt_boxes[i_gt, 2] - rpn_simu_gt_boxes[i_gt, 0],
                          rpn_simu_gt_boxes[i_gt, 3] - rpn_simu_gt_boxes[i_gt, 1], fill=False,
                          edgecolor='red', linewidth=3.5)
        )
        print '>>>>>>>>>>>>>>>>>>>>>>>>',rpn_simu_gt_boxes[i_gt, :5], img_info
        '''
        vis_detect(img_data.copy(), gt_box, simu_point, rpn_boxes_filtered, averge_boxes, flipped,
                   hyper_rpn_conv_features, merged_proposal,
                   view_detect)
    return rpn_simu_gt_boxes_average, rpn_simu_gt_boxes_feature_map


def vis_detect(img_data, gt_box, simu_point, temp_filtered, averge_boxes, flipped, hyper_rpn_conv_features,
               merged_proposal, isshow):
    if isshow:
        # temp_filtered = temp_filtered[0]
        img_data_shape = np.shape(img_data)
        # print np.shape(img_data)
        img_data = img_data.reshape((img_data_shape[1], img_data_shape[2], img_data_shape[3]))
        # print np.shape(img_data)
        img_data = img_data.transpose((1, 2, 0))
        img_data += cfg.PIXEL_MEANS
        img_data = img_data[:, :, (2, 1, 0)]
        # print np.shape(img_data)
        # plt.figure(1)
        fig, ax = plt.subplots(figsize=(12, 12))
        ax.imshow(img_data.astype(np.uint8), aspect='equal')
        ax.add_patch(
            plt.Rectangle((gt_box[0], gt_box[1]),
                          gt_box[2] - gt_box[0],
                          gt_box[3] - gt_box[1], fill=False,
                          edgecolor='green', linewidth=1.5)
        )
        plt.plot(simu_point[0], simu_point[1], marker='*', markersize=12)
        for rpn_boxes_filtered_i in temp_filtered:
            ax.add_patch(
                plt.Rectangle((rpn_boxes_filtered_i[0], rpn_boxes_filtered_i[1]),
                              rpn_boxes_filtered_i[2] - rpn_boxes_filtered_i[0],
                              rpn_boxes_filtered_i[3] - rpn_boxes_filtered_i[1], fill=False,
                              edgecolor='blue', linewidth=1)
            )
        ax.add_patch(
            plt.Rectangle((averge_boxes[0], averge_boxes[1]),
                          averge_boxes[2] - averge_boxes[0],
                          averge_boxes[3] - averge_boxes[1], fill=False,
                          edgecolor='red', linewidth=1)
        )
        merged = merged_proposal[0]
        ax.add_patch(
            plt.Rectangle((merged[0], merged[1]),
                          merged[2] - merged[0],
                          merged[3] - merged[1], fill=False,
                          edgecolor='white', linewidth=3)
        )
        plt.axis('off')
        plt.tight_layout()
        plt.title('small_object:' + str(int(simu_point[2])) + ' ,flipped:' + str(len(flipped)) +
                  ' ,rpn_filterd_boxes:' + str(len(temp_filtered)))
        plt.draw()

        # print 'hyper_rpn_conv_features'
        featur_shape = np.shape(hyper_rpn_conv_features)
        hyper_rpn_conv_features = hyper_rpn_conv_features.reshape((featur_shape[1], featur_shape[2], featur_shape[3]))
        fig2, ax2 = plt.subplots(figsize=(12, 12))
        gci = ax2.imshow(np.average(hyper_rpn_conv_features, axis=0))
        plt.colorbar(gci, fraction=0.046, pad=0.04)
        for rpn_boxes_filtered_i in temp_filtered:
            rpn_boxes_filtered_i = rpn_boxes_filtered_i / 4
            ax2.add_patch(
                plt.Rectangle((rpn_boxes_filtered_i[0], rpn_boxes_filtered_i[1]),
                              rpn_boxes_filtered_i[2] - rpn_boxes_filtered_i[0],
                              rpn_boxes_filtered_i[3] - rpn_boxes_filtered_i[1], fill=False,
                              edgecolor='blue', linewidth=1)
            )
        plt.title('hyper_rpn_conv_features')
        plt.show()


def get_grid_bbox(height, width):
    grid_pixels = 10  # im.shape (rows,columns,channel) with respect to (height,width)
    grid_height_num = height / grid_pixels if height % grid_pixels == 0 else height / grid_pixels + 1
    grid_width_num = width / grid_pixels if width % grid_pixels == 0 else width / grid_pixels + 1
    img_grids = np.zeros((grid_height_num * grid_width_num, 4), dtype=np.uint16)
    index = 0
    for h in range(0, height, grid_pixels):
        for w in range(0, width, grid_pixels):
            img_grids[index] = [w, h,
                                w + grid_pixels if w <= width and width - w > grid_pixels else width - 1,
                                h + grid_pixels if h <= height and height - h > grid_pixels
                                else height - 1]
            index += 1
    return img_grids


def get_grids(height, width, starty, startx, feature_shape, is_small_object):
    print height, width, starty, startx
    feature_height = feature_shape[2]
    feature_width = feature_shape[3]
    extended_pixels = 5
    if is_small_object:
        grid_number = 10
    else:
        grid_number = 13
    startx = startx - extended_pixels / 2 if startx - extended_pixels / 2 > 0 else 0
    starty = starty - extended_pixels / 2 if starty - extended_pixels / 2 > 0 else 0
    width = width + extended_pixels if width + extended_pixels < feature_width else feature_width
    height = height + extended_pixels if height + extended_pixels < feature_height else feature_height
    print height, width, starty, startx
    x = np.linspace(startx, startx + width, grid_number + 1)
    y = np.linspace(starty, starty + height, grid_number + 1)
    xv, yv = np.meshgrid(x, y)
    xv1 = np.delete(np.delete(xv, -1, axis=1), -1, axis=0)
    yv1 = np.delete(np.delete(yv, -1, axis=0), -1, axis=1)
    xv2 = np.delete(np.delete(xv, 0, axis=1), 0, axis=0)
    yv2 = np.delete(np.delete(yv, 0, axis=0), 0, axis=1)
    merged = np.round(np.hstack((np.hstack((xv1.reshape((grid_number ** 2, 1)), yv1.reshape((grid_number ** 2, 1)))),
                                 np.hstack((xv2.reshape((grid_number ** 2, 1)),
                                            yv2.reshape((grid_number ** 2, 1))))))).astype(np.uint16)
    return merged


def get_distance(x_a, y_a, x_b, y_b, x_c, y_c):
    if int(x_b) is not 0 and int(y_b) is not 0:
        # the distance of a point to a line
        vector_a_b = np.array([x_b - x_a, y_b - y_a])
        vector_a_c = np.array([x_c - x_a, y_c - y_a])
        vector_c = np.dot(vector_a_b, vector_a_c) * vector_a_b / np.square(np.linalg.norm(vector_a_b))
        return np.linalg.norm(vector_a_c - vector_c)
    else:
        # the distance between 2 points
        return np.linalg.norm([x_c - x_a, y_c - y_a])
