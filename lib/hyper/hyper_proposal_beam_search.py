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
import matplotlib.pyplot as plt
from utils.cython_bbox import bbox_overlaps

DEBUG = False


class ProposalBeamSearch(caffe.Layer):
    """
    Outputs object detection proposals by applying estimated bounding-box
    transformations to a set of regular boxes (called "anchors").
    """

    def setup(self, bottom, top):
        # parse the layer parameter string, which must be valid YAML
        layer_params = yaml.load(self.param_str_)

        self._feat_stride = layer_params['feat_stride']  # feat_stride =4
        anchor_scales = layer_params.get('scales', (8, 16, 32, 64))
        self._anchors = generate_anchors(scales=np.array(anchor_scales))
        self._num_anchors = self._anchors.shape[0]
        if DEBUG:
            print 'feat_stride: {}'.format(self._feat_stride)
        top[0].reshape(1, 5)
        top[1].reshape(1, 1)

    def forward(self, bottom, top):
        # assert bottom[0].data.shape[0] == 1, \
        #     'Only single item batches are supported'

        simu_points = bottom[0].data
        flipped = bottom[1].data
        img_index = bottom[2].data
        im_info = bottom[3].data[0, :]
        data = bottom[4].data
        gt_boxes = bottom[5].data
        # hyper_features = bottom[6].data
        scores = bottom[6].data[:, self._num_anchors:, :, :]
        cfg_key = str(self.phase)  # either 'TRAIN' or 'TEST'
        pre_nms_topN = cfg[
            cfg_key].RPN_PRE_NMS_TOP_N  # 12000 Number of top scoring boxes to keep before apply NMS to RPN proposals
        post_nms_topN = cfg[
            cfg_key].RPN_POST_NMS_TOP_N  # 2000 Number of top scoring boxes to keep after applying NMS to RPN proposals
        nms_thresh = cfg[cfg_key].RPN_NMS_THRESH  # 0.7 NMS threshold used on RPN proposals
        min_size = cfg[
            cfg_key].RPN_MIN_SIZE  # 16 Proposal height and width both need to be greater than RPN_MIN_SIZE (at orig image scale)
        if DEBUG:
            print 'im_size: ({}, {})'.format(im_info[0], im_info[1])
            print 'scale: {}'.format(im_info[2])
            # print 'feature_shape: {}'.format(np.shape(hyper_features))
        # feature_shape = np.shape(hyper_features)
        # hyper_features = hyper_features.reshape((feature_shape[1], feature_shape[2], feature_shape[3]))
        # hyper_proposals = np.zeros((len(gt_boxes) * 7, 4))
        # hyper_labels = np.zeros((len(gt_boxes) * 7,))

        height, width = scores.shape[-2:]

        if DEBUG:
            print 'score map size: {}'.format(scores.shape)
            print '=============================================='
            print 'height,width:({},{})'.format(height, width)
            print '=============================================='
        # Enumerate all shifts
        shift_x = np.arange(0, width) * self._feat_stride
        shift_y = np.arange(0, height) * self._feat_stride
        shift_x, shift_y = np.meshgrid(shift_x, shift_y)
        shifts = np.vstack((shift_x.ravel(), shift_y.ravel(),
                            shift_x.ravel(), shift_y.ravel())).transpose()
        A = self._num_anchors
        K = shifts.shape[0]
        anchors = self._anchors.reshape((1, A, 4)) + \
                  shifts.reshape((1, K, 4)).transpose((1, 0, 2))
        anchors = anchors.reshape((K * A, 4))
        scores = scores.transpose((0, 2, 3, 1)).reshape((-1, 1))
        proposals = anchors.astype(np.float, copy=False)
        if DEBUG:
            print 'num anchors:{}'.format(np.shape(anchors))
            print 'num proposals:{}'.format(np.shape(proposals))
        proposals = clip_boxes(proposals, im_info[:2])
        keep = _filter_boxes(proposals, min_size * im_info[2])
        proposals = proposals[keep, :]
        scores = scores[keep]
        order = scores.ravel().argsort()[::-1]
        if pre_nms_topN > 0:
            order = order[:30000]
        proposals = proposals[order, :]
        scores = scores[order]
        # print 'proposals_num_before nums:{}'.format(len(proposals))
        keep = nms(np.hstack((proposals.astype(np.float32), scores.astype(np.float32))), 0.8)
        if post_nms_topN > 0:
            keep = keep[:post_nms_topN]
        proposals = proposals[keep, :]
        scores = scores[keep]
        # print 'proposals_num_after nums:{}'.format(len(proposals))
        # print 'len(gt_boxes):{},{}'.format(len(gt_boxes), gt_boxes)
        total_boxes_per_image = 128
        hyper_pro_each = total_boxes_per_image / len(gt_boxes)
        hyper_proposal_number = hyper_pro_each * len(gt_boxes)
        hyper_proposals = np.zeros((hyper_proposal_number + len(gt_boxes), 4))
        hyper_labels = np.zeros((hyper_proposal_number + len(gt_boxes),))

        index = 0
        for gt_box in gt_boxes:
            label = gt_box[4] - 1
            # gt_box = gt_box[0:4] / self._feat_stride  # feature map size
            overlaps = bbox_overlaps(proposals.astype(np.float),
                                     np.array(gt_box[0:4]).reshape((1, 4)).astype(np.float))
            thresh = 0.6

            keep = np.where(np.array(overlaps) >= thresh)[0]
            while len(keep) < hyper_pro_each and thresh > 0.4:
                thresh -= 0.05
                keep = np.where(np.array(overlaps) >= thresh)[0]
            if thresh > 0.4:
                proposals_ = proposals[keep, :]
                scores_ = scores[keep, :]

                order = scores_.ravel().argsort()[::-1]
                order = order[:hyper_pro_each]

                proposals_ = proposals_[order, :]
                # print 'len(proposals_)"{}:,hyper_pro_each:{}'.format(len(proposals_), hyper_pro_each)
                scores_ = scores_[order]
                hyper_proposals[index, :] = np.array(gt_box[0:4]).reshape((1, 4))
                hyper_labels[index] = label
                # print '_____', gt_box, hyper_proposals[index, :], label
                index += 1
                hyper_proposals[index:index + hyper_pro_each, :] = proposals_[:hyper_pro_each]
                hyper_labels[index:index + hyper_pro_each] = label
                index += hyper_pro_each
            else:
                hyper_proposals[index, :] = np.array(gt_box[0:4]).reshape((1, 4))
                hyper_labels[index] = label
                # print '_____', gt_box, hyper_proposals[index, :], label
                index += 1
                hyper_proposals[index:index + hyper_pro_each, :] = np.array(gt_box[0:4]).reshape((1, 4))
                hyper_labels[index:index + hyper_pro_each] = label
                index += hyper_pro_each
            keep = np.where(np.array(overlaps) < thresh)[0]
            proposals = proposals[keep, :]
            scores = scores[keep, :]
        # print hyper_labels, hyper_proposals
        # for index, gt_box in enumerate(gt_boxes):
        #     label = gt_box[4] - 1  # we don't have background class, so we set label minus 1
        #     gt_box = gt_box[:4] / self._feat_stride  # feature map size
        #     width = gt_box[2] - gt_box[0]
        #     height = gt_box[3] - gt_box[1]
        #     startx = int(gt_box[0])
        #     starty = int(gt_box[1])
        #     grids = get_grids(int(height), int(width), starty, startx, feature_shape)
        #
        #     # (chanel,height,width)
        #     grid_value = [np.sum(hyper_features[:, grid[1]:grid[3], grid[0]:grid[2]]) for grid in grids]
        #     max_ = np.max(grid_value)
        #     # sum_ = np.sum(np.exp(grid_value - max_))
        #     # grid_value = np.exp(grid_value - max_) / sum_
        #     sum_ = np.sum(grid_value)
        #     grid_value = grid_value / sum_
        #     average_value = np.average(grid_value)
        #     # order = np.array(grid_value).ravel().argsort()[::-1]
        #     order = np.where(grid_value >= average_value)[0]
        #     # if is_small_object == 1:
        #     #     order = order[:20]
        #     # else:
        #     #     order = order[:40]
        #     grids = np.array(grids)[order, :]
        #     target_proposal = [np.min(grids[:, 0]), np.min(grids[:, 1]), np.max(grids[:, 2]), np.max(grids[:, 3])]
        #     proposals = gernerater_beam_serarch_proposals(target_proposal, feature_shape)
        #     # print np.shape(hyper_proposals), index, np.shape(hyper_labels), label
        #     hyper_proposals[index * 7:index * 7 + 7, :] = proposals
        #     hyper_labels[index * 7:index * 7 + 7] = label
        #
        #     # fig2, ax2 = plt.subplots(figsize=(12, 12))
        #     # gci = ax2.imshow(np.average(hyper_features, axis=0))
        #     # plt.colorbar(gci, fraction=0.046, pad=0.04)
        #     # for grid in grids:
        #     #     ax2.add_patch(
        #     #         plt.Rectangle((grid[0], grid[1]),
        #     #                       grid[2] - grid[0],
        #     #                       grid[3] - grid[1], fill=False,
        #     #                       edgecolor='yellow', linewidth=0.5)
        #     #     )
        #     # for p in proposals:
        #     #     ax2.add_patch(
        #     #
        #     #         plt.Rectangle((p[0], p[1]),
        #     #                       p[2] - p[0],
        #     #                       p[3] - p[1], fill=False,
        #     #                       edgecolor='white', linewidth=1)
        #     #     )
        #     # print proposals
        #     # plt.title('hyper_rpn_conv_features')
        #     # plt.show()

        batch_inds = np.zeros((hyper_proposals.shape[0], 1), dtype=np.float32)
        # hyper_proposals = hyper_proposals/im_info[2]
        blob = np.hstack((batch_inds, hyper_proposals.astype(np.float32, copy=False) / self._feat_stride))
        # print 'hyper_proposal', np.shape(hyper_labels), np.shape(blob),hyper_labels
        top[0].reshape(*(blob.shape))
        top[0].data[...] = blob
        top[1].reshape(*(hyper_labels.shape))
        top[1].data[...] = hyper_labels

    def backward(self, top, propagate_down, bottom):
        """This layer does not propagate gradients."""
        pass

    def reshape(self, bottom, top):
        """Reshaping happens during the call to forward."""
        pass


def _filter_boxes(boxes, min_size):
    """Remove all boxes with any side smaller than min_size."""
    ws = boxes[:, 2] - boxes[:, 0] + 1
    hs = boxes[:, 3] - boxes[:, 1] + 1
    keep = np.where((ws >= min_size) & (hs >= min_size))[0]
    return keep


def gernerater_beam_serarch_proposals(target_proposal, feature_shape):
    all_proposals = []
    feature_height = feature_shape[2]
    feature_width = feature_shape[3]
    zoom_in_ratio = 1.1
    zoom_out_ratio = 0.9
    moved_dis = 5
    target_center_x = (target_proposal[0] + target_proposal[2]) / 2
    target_center_y = (target_proposal[1] + target_proposal[3]) / 2
    # zoom in target_proposal
    trans_dis_x = target_center_x * (zoom_in_ratio - 1)
    trans_dis_y = target_center_y * (zoom_in_ratio - 1)
    temp = np.array(target_proposal) * zoom_in_ratio
    proposal = [temp[0] - trans_dis_x, temp[1] - trans_dis_y,
                temp[2] - trans_dis_x, temp[3] - trans_dis_y]
    all_proposals.append(proposal)
    # zoom out target_proposal
    trans_dis_x = target_center_x * (1 - zoom_out_ratio)
    trans_dis_y = target_center_y * (1 - zoom_out_ratio)
    temp = np.array(target_proposal) * zoom_out_ratio
    proposal = [temp[0] + trans_dis_x, temp[1] + trans_dis_y,
                temp[2] + trans_dis_x, temp[3] + trans_dis_y]
    all_proposals.append(proposal)
    # move target_proposal to right
    all_proposals.append([target_proposal[0] + moved_dis, target_proposal[1],
                          target_proposal[2] + moved_dis, target_proposal[3]])
    # move target_proposal to left
    all_proposals.append([target_proposal[0] - moved_dis, target_proposal[1],
                          target_proposal[2] - moved_dis, target_proposal[3]])
    # move target_proposal to up
    all_proposals.append([target_proposal[0], target_proposal[1] - moved_dis,
                          target_proposal[2], target_proposal[3] - moved_dis])
    # move target_proposal to down
    all_proposals.append([target_proposal[0], target_proposal[1] + moved_dis,
                          target_proposal[2], target_proposal[3] + moved_dis])
    all_proposals.append(target_proposal)
    return check_sanity(all_proposals, feature_height, feature_width)


def check_sanity(all_proposals, height, width):
    checked = []
    for p in all_proposals:
        checked.append([p[0] if p[0] > 0 else 0,
                        p[1] if p[1] > 0 else 0,
                        p[2] if ((p[2] - p[0]) < width and p[0] >= 0) or (p[2] < width and p[0] < 0)  else width,
                        p[3] if ((p[3] - p[1]) < height and p[3] >= 0) or (p[3] < height and p[1] < 0) else height])

    return np.array(checked).reshape((len(checked), 4))


def get_grids(height, width, starty, startx, feature_shape):
    # print height, width, starty, startx
    feature_height = feature_shape[2]
    feature_width = feature_shape[3]
    extended_pixels = 0
    grid_number = 13
    startx = startx - extended_pixels / 2 if startx - extended_pixels / 2 > 0 else 0
    starty = starty - extended_pixels / 2 if starty - extended_pixels / 2 > 0 else 0
    width = width + extended_pixels if width + extended_pixels < feature_width else feature_width
    height = height + extended_pixels if height + extended_pixels < feature_height else feature_height
    # print height, width, starty, startx
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
