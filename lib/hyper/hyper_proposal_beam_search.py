# --------------------------------------------------------
# Faster R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick and Sean Bell
# --------------------------------------------------------

import caffe
import numpy as np
import numpy.random as npr
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
        # hyper_features = bottom[7].data
        cfg_key = str(self.phase)  # either 'TRAIN' or 'TEST'
        pre_nms_topN = cfg[
            cfg_key].RPN_PRE_NMS_TOP_N  # 12000 Number of top scoring boxes to keep before apply NMS to RPN proposals
        post_nms_topN = cfg[
            cfg_key].RPN_POST_NMS_TOP_N  # 2000 Number of top scoring boxes to keep after applying NMS to RPN proposals
        nms_thresh = cfg[cfg_key].RPN_NMS_THRESH  # 0.7 NMS threshold used on RPN proposals
        min_size = cfg[
            cfg_key].RPN_MIN_SIZE  # 16 Proposal height and width both need to be greater than RPN_MIN_SIZE (at orig image scale)
        # feature_shape = np.shape(hyper_features)
        # hyper_features = hyper_features.reshape((feature_shape[1], feature_shape[2], feature_shape[3]))
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
        scores_ori = scores.transpose((0, 2, 3, 1)).reshape((-1, 1))
        proposals_ori = anchors.astype(np.float, copy=False)

        if DEBUG:
            print 'num anchors:{}'.format(np.shape(anchors))
            print 'num proposals_ori:{}'.format(np.shape(proposals_ori))
        proposals_ori = clip_boxes(proposals_ori, im_info[:2])
        keep = _filter_boxes(proposals_ori, min_size * im_info[2])

        proposals_ori = proposals_ori[keep, :]
        scores_ori = scores_ori[keep]

        index = 0
        fg_fraction = 0.5  # positive vs negative =1
        fg_boxes_per_image = 64
        bg_boxes_per_image = 64
        fg_pro_each = fg_boxes_per_image / len(gt_boxes)
        fg_pro_num = fg_pro_each * len(gt_boxes)
        bg_pro_each = bg_boxes_per_image / len(gt_boxes)
        bg_pro_num = bg_pro_each * len(gt_boxes)
        hyper_proposals = np.zeros((fg_pro_num + bg_pro_num + len(gt_boxes), 4))
        hyper_labels = np.zeros((fg_pro_num + bg_pro_num + len(gt_boxes),))

        for gt_box in gt_boxes:
            label = gt_box[4]
            overlaps = bbox_overlaps(proposals_ori.astype(np.float),
                                     np.array(gt_box[0:4]).reshape((1, 4)).astype(np.float))
            thresh = 0.6
            keep = np.where(np.array(overlaps) >= thresh)[0]
            while len(keep) < fg_pro_each and thresh > 0.4:
                thresh -= 0.05
                keep = np.where(np.array(overlaps) >= thresh)[0]
            proposals = proposals_ori[keep, :]
            scores = scores_ori[keep]

            order = scores.ravel().argsort()[::-1]
            if pre_nms_topN > 0:
                order = order[:pre_nms_topN]
            proposals = proposals[order, :]
            scores = scores[order]

            keep = nms(np.hstack((proposals.astype(np.float32), scores.astype(np.float32))), nms_thresh)
            while len(keep) < fg_pro_each and nms_thresh < 0.9:
                nms_thresh += 0.05
                keep = nms(np.hstack((proposals.astype(np.float32), scores.astype(np.float32))), nms_thresh)
            nms_thresh = 0.7
            proposals = proposals[keep, :]
            scores = scores[keep]

            # fig2, ax2 = plt.subplots(figsize=(12, 12))
            # gci = ax2.imshow(np.average(hyper_features, axis=0))
            # plt.colorbar(gci, fraction=0.046, pad=0.04)
            # for p in proposals:
            #     p = p / self._feat_stride
            #     ax2.add_patch(
            #
            #         plt.Rectangle((p[0], p[1]),
            #                       p[2] - p[0],
            #                       p[3] - p[1], fill=False,
            #                       edgecolor='white', linewidth=1)
            #     )
            # plt.title('hyper_rpn_conv_features')
            # plt.show()

            proposals_ = proposals
            scores_ = scores
            order = scores_.ravel().argsort()[::-1]
            order = order[:fg_pro_each]
            proposals_ = proposals_[order, :]

            # positive examples
            hyper_proposals[index, :] = np.array(gt_box[0:4]).reshape((1, 4))
            hyper_labels[index] = label
            index += 1
            if len(proposals_) >= fg_pro_each:
                hyper_proposals[index:index + fg_pro_each, :] = proposals_[:fg_pro_each]
            else:
                hyper_proposals[index:index + len(proposals_), :] = proposals_[:len(proposals_)]
                hyper_proposals[index + len(proposals_):fg_pro_each, :] = np.array(gt_box[0:4]).reshape((1, 4))
            hyper_labels[index:index + fg_pro_each] = label
            index += fg_pro_each

            # negative examples
            keep = np.where((np.array(overlaps) > 0.1) & (np.array(overlaps) < 0.3))[0]
            # print 'len(negative):', len(keep)
            bg_proposals = proposals_ori[keep, :]
            if len(keep) >= bg_pro_each:
                hyper_proposals[index:index + bg_pro_each, :] = bg_proposals[:bg_pro_each]
            else:
                iou_thresh = 0.35
                while len(keep) < bg_pro_each and iou_thresh < 0.5:
                    keep = np.where((np.array(overlaps) > 0.1) & (np.array(overlaps) < iou_thresh))[0]
                    iou_thresh += 0.05
                if len(keep) < bg_pro_each:
                    bg_proposals = proposals_ori[0:bg_pro_each, :]
                else:
                    bg_proposals = proposals_ori[keep, :]

                print 'len(bg_proposals):', len(bg_proposals), iou_thresh
                hyper_proposals[index:index + bg_pro_each, :] = bg_proposals[:bg_pro_each]
            hyper_labels[index:index + bg_pro_each] = 0  # set 0 as background
            index += bg_pro_each
            keep = np.where(np.array(overlaps) < thresh)[0]
            proposals_ori = proposals_ori[keep, :]
            scores_ori = scores_ori[keep, :]

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
        blob = np.hstack((batch_inds, hyper_proposals.astype(np.float32, copy=False) / self._feat_stride))
        # print 'hyper_proposal', hyper_proposals/ self._feat_stride,  hyper_labels
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


def _sample_negative_rois(all_rois, gt_boxes, fg_rois_per_image, rois_per_image, ):
    """Generate a random sample of RoIs comprising foreground and background
    examples.
    """
    # overlaps: (rois x gt_boxes)
    overlaps = bbox_overlaps(
        np.ascontiguousarray(all_rois[:, 1:5], dtype=np.float),
        np.ascontiguousarray(gt_boxes[:, :4], dtype=np.float))
    gt_assignment = overlaps.argmax(axis=1)
    max_overlaps = overlaps.max(axis=1)
    labels = gt_boxes[gt_assignment, 4]

    # Select foreground RoIs as those with >= FG_THRESH overlap
    fg_inds = np.where(max_overlaps >= cfg.TRAIN.FG_THRESH)[0]
    # Guard against the case when an image has fewer than fg_rois_per_image
    # foreground RoIs
    fg_rois_per_this_image = min(fg_rois_per_image, fg_inds.size)
    fg_rois_per_this_image = int(fg_rois_per_this_image)
    # Sample foreground regions without replacement
    if fg_inds.size > 0:
        fg_inds = npr.choice(fg_inds, size=fg_rois_per_this_image, replace=False)

    # Select background RoIs as those within [BG_THRESH_LO, BG_THRESH_HI)
    bg_inds = np.where((max_overlaps < cfg.TRAIN.BG_THRESH_HI) &
                       (max_overlaps >= cfg.TRAIN.BG_THRESH_LO))[0]
    # Compute number of background RoIs to take from this image (guarding
    # against there being fewer than desired)
    bg_rois_per_this_image = rois_per_image - fg_rois_per_this_image
    bg_rois_per_this_image = min(bg_rois_per_this_image, bg_inds.size)
    # Sample background regions without replacement
    if bg_inds.size > 0:
        bg_inds = npr.choice(bg_inds, size=bg_rois_per_this_image, replace=False)

    # The indices that we're selecting (both fg and bg)
    keep_inds = np.append(fg_inds, bg_inds)
    # Select sampled values from various arrays:
    labels = labels[keep_inds]
    # Clamp labels for the background RoIs to 0
    labels[fg_rois_per_this_image:] = 0
    rois = all_rois[keep_inds]
    return labels, rois,
