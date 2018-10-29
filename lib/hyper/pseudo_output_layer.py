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
DEBUG = True


class PseudoOutputLayer(caffe.Layer):
    """
    Outputs object detection proposals by applying estimated bounding-box
    transformations to a set of regular boxes (called "anchors").
    """
    iter_times = 0
    train_times = 1000  # each 1000 times to calculate train accuracy
    total_objects = 0
    total_postives = 0
    pse_gt_list = []

    def setup(self, bottom, top):
        # parse the layer parameter string, which must be valid YAML
        layer_params = yaml.load(self.param_str_)
        self.feat_stride = layer_params['feat_stride']  # feat_stride =4
        if DEBUG:
            print 'feat_stride: {:}'.format(self.feat_stride)
            # rois blob: holds R regions of interest, each is a 5-tuple
            # (n, x1, y1, x2, y2) specifying an image batch index n and a
            # rectangle (x1, y1, x2, y2)
        top[0].reshape(1, 5)

    def forward(self, bottom, top):
        hyper_rois = bottom[0].data
        hyper_cls_score = bottom[1].data
        hyper_pool = bottom[2].data
        hyper_labels = bottom[3].data
        gt_boxes = bottom[4].data
        flipped = bottom[5].data
        img_index = bottom[6].data
        im_info = bottom[7].data[0]
        one_epoc = 10022
        if DEBUG and self.iter_times % 200 == 0:
            print '+++++++++++++++++++++++++++++++++++++++++++++++++++++'
            print 'hyper_rois: {},{},{}'.format(len(hyper_rois), np.shape(hyper_rois),hyper_rois)
            print 'hyper_cls_score:{},{}'.format(np.shape(hyper_cls_score),
                                                 np.argmax(hyper_cls_score, axis=1))
            print 'hyper_labels:{}'.format(hyper_labels)
            print 'pred_labels:{}'.format(np.argmax(hyper_cls_score, axis=1))
            print 'hyper_pool:{}'.format(np.shape(hyper_pool))
            # print 'roi data labels:{},{}'.format(len(labels),np.shape(labels))
            print '+++++++++++++++++++++++++++++++++++++++++++++++++++++'
        per_image_classes = len(gt_boxes)
        hyper_pro_each = 128 / per_image_classes + 1
        hyper_rois_ = hyper_rois[::hyper_pro_each, 1:5] * self.feat_stride
        hyper_labels_ = hyper_labels[::hyper_pro_each].reshape(
            (len(hyper_labels) / hyper_pro_each, 1)) + 1  # plus one as true label
        self.iter_times += 1
        self.total_objects += len(hyper_rois)
        pred_labels = np.argmax(hyper_cls_score, axis=1)
        self.total_postives += np.shape(np.equal(hyper_labels.astype(np.int16), pred_labels).nonzero())[1]
        train_accuracy = 1.0 * self.total_postives / self.total_objects
        if self.iter_times % self.train_times == 0:
            print '~~~~~~~~~~~~~'
            print 'train_accuracy:{},{},{}'.format(self.total_postives, self.total_objects, train_accuracy)
            print '~~~~~~~~~~~~~'
            self.total_objects = 0
            self.total_postives = 0

        if train_accuracy > 0.7 and self.iter_times > 70000:
            for index, gt_box in enumerate(gt_boxes):
                hyper_cls_score_ = hyper_cls_score[index * hyper_pro_each:(index + 1) * hyper_pro_each]
                rois_ = hyper_rois[index * hyper_pro_each:(index + 1) * hyper_pro_each, :]
                pred_label_ = pred_labels[index * hyper_pro_each:(index + 1) * hyper_pro_each]
                label = int(gt_box[4]) - 1
                gt_score = hyper_cls_score_[0, label]
                keep = np.where((hyper_cls_score_[:, label]) > gt_score & (pred_label_ == label))[0]
                # print 'hyper_cls_score_:', len(hyper_cls_score_), hyper_cls_score_
                # print 'hyper_rois_', len(rois_), rois_
                # print 'gt_score:', gt_score
                # print 'keep:', keep

                if len(keep) > 1:
                    # print '******************* find one   *****************'
                    # print self.iter_times, "___________________"
                    rois_ = rois_[keep, :]
                    hyper_cls_score_ = hyper_cls_score_[keep, :]

                    keep = hyper_cls_score_.ravel().argsort()[::-1]
                    rois_ = rois_[keep, :]
                    hyper_cls_score_ = hyper_cls_score_[keep, :]

                    print '________hyper_cls_score________'
                    print hyper_cls_score_
                    print '________hyper_cls_score________'
                    best_score = hyper_cls_score_[0]
                    best_roi = rois_[0]
                    for i, rois_i in enumerate(rois_):
                        # if highest score bounding boxes contained another one, don't consider this one
                        if (best_roi[1] <= rois_i[1] and best_roi[2] <= rois_i[2]
                            and best_roi[3] >= rois_i[3] and best_roi[4] >= rois_i[4]):
                            if np.abs(best_score / hyper_cls_score_[i]) <= 1.05:
                                best_score = hyper_cls_score_[i]
                                best_roi = rois_i
                    hyper_rois_[index, :] = best_roi[1:5] * self.feat_stride
                    # print hyper_rois_[index, :], rois_[keep[0], :]
                else:
                    hyper_rois_[index, :] = rois_[keep[0], 1:5] * self.feat_stride
        if len(self.pse_gt_list) < one_epoc:
            self.pse_gt_list.append({'img_index': int(img_index),
                                     'flipped': int(flipped),
                                     'rpn_simu_gt_boxes': hyper_rois_ / im_info[2]})
        if len(self.pse_gt_list) == one_epoc:
            if self.iter_times == one_epoc:
                self.pse_gt_list.sort(key=take_key)
            elif self.iter_times > one_epoc:
                save_pse_results(self, int(img_index), int(flipped), hyper_rois_ / im_info[2], one_epoc,
                                 'classifi_', self.iter_times)
        # print 'hyper_rois',hyper_rois, hyper_labels
        hyper_gt_boxes = np.concatenate((hyper_rois_, hyper_labels_), axis=1)
        # print hyper_gt_boxes
        top[0].reshape(*hyper_gt_boxes.shape)
        top[0].data[...] = hyper_gt_boxes

    def backward(self, top, propagate_down, bottom):
        """This layer does not propagate gradients."""
        pass

    def reshape(self, bottom, top):
        """Reshaping happens during the call to forward."""
        pass


def take_key(elem):
    return elem['img_index']


def save_pse_results(self, img_index, flipped, rpn_simu_gt_boxes, one_epoc, name, iter_times):
    order = [rpn_boxes['img_index'] for rpn_boxes in self.pse_gt_list]
    i_dx = order.index(int(img_index))
    temp1 = self.pse_gt_list[i_dx]
    temp2 = self.pse_gt_list[i_dx + 1]
    if temp1['flipped'] == flipped:
        self.pse_gt_list[i_dx] = {'img_index': int(img_index),
                                  'flipped': int(flipped),
                                  'rpn_simu_gt_boxes': rpn_simu_gt_boxes}
    elif temp2['flipped'] == flipped:
        self.pse_gt_list[i_dx + 1] = {'img_index': int(img_index),
                                      'flipped': int(flipped),
                                      'rpn_simu_gt_boxes': rpn_simu_gt_boxes}
    if (iter_times - 1) % one_epoc == 0:
        cache_file = os.path.join(cache_path +
                                  'voc_2007_trainval_rpn_pse_gt_boxes_' + str(name) + str(
            iter_times + 1) + '.pkl')
        with open(cache_file, 'wb') as fid:
            cPickle.dump(self.pse_gt_list, fid, cPickle.HIGHEST_PROTOCOL)


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
