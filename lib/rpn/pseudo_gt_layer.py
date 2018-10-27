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
        self.rpn_pse_gt_list = []
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
        if DEBUG:
            print '+++++++++++++++++++++++++++++++++++++++++++++++++++++'
            print 'rpn_rois: ({}, {},{})'.format(len(rpn_rois), rpn_rois[0], np.shape(rpn_rois))
            print 'simu_points: {},{}'.format(len(simu_points), simu_points)
            print 'flipped:{}'.format(flipped)
            print 'img_index:{}'.format(img_index)
            print 'img_info:{}'.format(img_info)
            print '+++++++++++++++++++++++++++++++++++++++++++++++++++++'
        rpn_boxes = rpn_rois[:, 1:5]
        one_epoc = 10022
        rpn_simu_gt_boxes = generate_rpn_simu_gt_boxes(img_data, simu_points, gt_boxes, img_info, rpn_boxes, flipped)
        # print 'rpn_simu_gt_boxes:{}'.format(rpn_simu_gt_boxes)
        if len(self.rpn_pse_gt_list) < one_epoc:
            self.rpn_pse_gt_list.append({'img_index': int(img_index),
                                         'rpn_simu_gt_boxes': rpn_simu_gt_boxes})
        if len(self.rpn_pse_gt_list) == one_epoc:
            if self.iter_times + 1 == one_epoc:
                self.rpn_pse_gt_list.sort(key=take_key)
                # cache_file = os.path.join(cache_path +
                #                           'voc_2007_trainval_rpn_pse_gt_boxes_compare_' + str(
                #     self.iter_times + 1) + '.pkl')
                # with open(cache_file, 'wb') as fid:
                #     cPickle.dump(self.rpn_pse_gt_list, fid, cPickle.HIGHEST_PROTOCOL)
            elif self.iter_times + 1 > one_epoc:
                order = [rpn_boxes['img_index'] for rpn_boxes in self.rpn_pse_gt_list]
                i_dx = order.index(int(img_index))
                temp1 = self.rpn_pse_gt_list[i_dx]
                temp2 = self.rpn_pse_gt_list[i_dx + 1]
                if (temp1['rpn_simu_gt_boxes'][:, 4] == rpn_simu_gt_boxes[0, 4]).all():
                    self.rpn_pse_gt_list[i_dx] = {'img_index': int(img_index),
                                                  'rpn_simu_gt_boxes': rpn_simu_gt_boxes}
                elif (temp2['rpn_simu_gt_boxes'][:, 4] == rpn_simu_gt_boxes[0, 4]).all():
                    self.rpn_pse_gt_list[i_dx + 1] = {'img_index': int(img_index),
                                                      'rpn_simu_gt_boxes': rpn_simu_gt_boxes}
                if self.iter_times % one_epoc == 0:
                    cache_file = os.path.join(cache_path +
                                              'voc_2007_trainval_rpn_pse_gt_boxes_comare_' + str(
                        self.iter_times + 1) + '.pkl')
                    with open(cache_file, 'wb') as fid:
                        cPickle.dump(self.rpn_pse_gt_list, fid, cPickle.HIGHEST_PROTOCOL)
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


def filter_rpn_boxes(rpn_boxes, simu_point, img_info, gt_box):
    margin = img_info[2] * 5
    filtered = []

    Top_N_Boxes_small = 5
    Top_N_Boxes_big = 10
    area_ratio_small = 0.1
    area_ratio_big_low = 0.1
    area_ratio_big_high = 0.85

    def take_elem(elem):
        return elem['sum_diff']

    while len(filtered) == 0:
        for box in rpn_boxes:
            if (box[0] - margin <= simu_point[0]) and (box[2] + margin >= simu_point[0]) and \
                    (box[1] - margin <= simu_point[1]) and (box[3] + margin >= simu_point[1]):
                distance_x = [np.abs(simu_point[0] - box[0]), np.abs(box[2] - simu_point[0])]
                distance_y = [np.abs(simu_point[1] - box[1]), np.abs(box[3] - simu_point[1])]
                diff_x = np.abs(np.max(distance_x) / np.min(distance_x))
                diff_y = np.abs(np.max(distance_y) / np.min(distance_y))
                sum_diff = diff_x + diff_y
                # filtered.append({'sum_diff': sum_diff,
                #                  'boxes': box})
                object_area = (box[2] - box[0]) * (box[3] - box[1]) / (img_info[0] * img_info[1])
                if int(simu_point[2]) == 1:  # small object

                    if object_area <= area_ratio_small:
                        filtered.append({'sum_diff': sum_diff,
                                         'boxes': box})
                else:
                    if object_area >= area_ratio_big_low and object_area <= area_ratio_big_high:
                        filtered.append({'sum_diff': sum_diff,
                                         'boxes': box})
        margin += img_info[2] * 5
    filtered.sort(key=take_elem)
    if int(simu_point[2]) == 1:  # small object
        if len(filtered) >= Top_N_Boxes_small:
            # print filtered[:Top_N_Boxes]
            return np.array([b['boxes'] for b in filtered[:Top_N_Boxes_small]]).reshape((Top_N_Boxes_small, 4))
        else:
            num_boxes = len(filtered)
            return np.array([b['boxes'] for b in filtered[:num_boxes]]).reshape((num_boxes, 4))
    else:
        if len(filtered) >= Top_N_Boxes_big:
            # print filtered[:Top_N_Boxes]
            return np.array([b['boxes'] for b in filtered[:Top_N_Boxes_big]]).reshape((Top_N_Boxes_big, 4))
        else:
            num_boxes = len(filtered)
            return np.array([b['boxes'] for b in filtered[:num_boxes]]).reshape((num_boxes, 4))


def generate_rpn_simu_gt_boxes(img_data, simu_points, gt_boxes, img_info, rpn_boxes, flipped):
    rpn_simu_gt_boxes = np.zeros((len(simu_points), 5))
    view_detect = False
    for i_gt in xrange(len(rpn_simu_gt_boxes)):
        iou_threshold = 0.7
        simu_point = simu_points[i_gt]
        gt_box = gt_boxes[i_gt]
        rpn_boxes_filtered = filter_rpn_boxes(rpn_boxes, simu_point, img_info, gt_box)
        overlaps = bbox_overlaps(rpn_boxes_filtered.astype(np.float),
                                 np.array(gt_box[0:4]).reshape((1, 4)).astype(np.float))
        # print 'len(rpn_boxes):', len(rpn_boxes)
        assert len(rpn_boxes_filtered) is not 0
        # rpn_boxes_filtered = rpn_boxes_filtered[np.where(np.array(overlaps) >= iou_threshold)[0], :]
        temp_filtered = rpn_boxes_filtered[np.where(np.array(overlaps) > iou_threshold)[0], :]
        while len(temp_filtered) == 0:
            temp_filtered = rpn_boxes_filtered[np.where(np.array(overlaps) > iou_threshold)[0], :]
            iou_threshold -= 0.05

        # if len(rpn_boxes_filtered) == 0:
        #     rpn_boxes_filtered = [box for box in rpn_boxes if
        #                           (box[0] - 5 <= simu_point[0]) and (box[2] + 5 >= simu_point[0]) and
        #                           (box[1] - 5 <= simu_point[1]) and (box[3] + 5 >= simu_point[1])]
        # print '>>>>>>>>>>>>>>', len(rpn_boxes), len(rpn_boxes_filtered), rpn_boxes_filtered[0]

        if view_detect:
            averge_boxes = np.average(temp_filtered, axis=0)  # / img_info[2]
        else:
            averge_boxes = np.average(temp_filtered, axis=0) / img_info[2]
        rpn_simu_gt_boxes[i_gt, :] = [averge_boxes[0], averge_boxes[1], averge_boxes[2], averge_boxes[3],
                                      int(flipped)]
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
        vis_detect(img_data.copy(), gt_box, simu_point, temp_filtered, rpn_boxes_filtered, averge_boxes, flipped,
                   view_detect)
    return rpn_simu_gt_boxes


def vis_detect(img_data, gt_box, simu_point, temp_filtered, rpn_boxes_filtered, averge_boxes, flipped, isshow):
    if isshow:
        import matplotlib.pyplot as plt
        # temp_filtered = temp_filtered[0]
        img_data_shape = np.shape(img_data)
        # print np.shape(img_data)
        img_data = img_data.reshape((img_data_shape[1], img_data_shape[2], img_data_shape[3]))
        # print np.shape(img_data)
        img_data = img_data.transpose((1, 2, 0))
        img_data += cfg.PIXEL_MEANS
        img_data = img_data[:, :, (2, 1, 0)]
        # print np.shape(img_data)
        fig, ax = plt.subplots(figsize=(12, 12))
        ax.imshow(img_data.astype(np.uint8), aspect='equal')
        ax.add_patch(
            plt.Rectangle((gt_box[0], gt_box[1]),
                          gt_box[2] - gt_box[0],
                          gt_box[3] - gt_box[1], fill=False,
                          edgecolor='green', linewidth=1.5)
        )
        plt.plot(simu_point[0], simu_point[1], marker='*', markersize=8)
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

        overlap_bbox_each = []
        for index_i, temp_i in enumerate(rpn_boxes_filtered):
            print 'first:', index_i, temp_i
            for _, temp_j in enumerate(rpn_boxes_filtered[index_i + 1:]):
                print 'second:', _, temp_j
                overlap_box = []
                if temp_i[0] <= temp_j[0]:
                    overlap_box.append(temp_j[0])
                else:
                    overlap_box.append(temp_i[0])
                if temp_i[1] <= temp_j[1]:
                    overlap_box.append(temp_j[1])
                else:
                    overlap_box.append(temp_i[1])
                if temp_i[2] <= temp_j[2]:
                    overlap_box.append(temp_i[2])
                else:
                    overlap_box.append(temp_j[2])
                if temp_i[3] <= temp_j[3]:
                    overlap_box.append(temp_i[3])
                else:
                    overlap_box.append(temp_j[3])
                # print overlap_box,len(overlap_box)
                # if len(overlap_box) is not 0:
                #     ax.add_patch(
                #         plt.Rectangle((overlap_box[0], overlap_box[1]),
                #                       overlap_box[2] - overlap_box[0],
                #                       overlap_box[3] - overlap_box[1], fill=False,
                #                       edgecolor='yellow', linewidth=1)
                #     )
                overlap_bbox_each.append(np.array(overlap_box))
        assert len(overlap_bbox_each) == len(rpn_boxes_filtered) * (len(rpn_boxes_filtered) - 1) / 2
        print np.array(overlap_bbox_each)
        overlap_average = np.average(np.array(overlap_bbox_each), axis=0)
        print overlap_average
        ax.add_patch(
            plt.Rectangle((overlap_average[0], overlap_average[1]),
                          overlap_average[2] - overlap_average[0],
                          overlap_average[3] - overlap_average[1], fill=False,
                          edgecolor='yellow', linewidth=1)
        )
        plt.axis('off')
        plt.tight_layout()
        plt.title('small_object:' + str(int(simu_point[2])) + ' ,flipped:' + str(len(flipped)) +
                  ' ,rpn_filterd_boxes:' + str(len(temp_filtered)))
        plt.draw()
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


def _filter_boxes(boxes, min_size):
    """Remove all boxes with any side smaller than min_size."""
    ws = boxes[:, 2] - boxes[:, 0] + 1
    hs = boxes[:, 3] - boxes[:, 1] + 1
    keep = np.where((ws >= min_size) & (hs >= min_size))[0]
    return keep
