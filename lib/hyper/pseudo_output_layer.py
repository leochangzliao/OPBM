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

    def setup(self, bottom, top):
        # parse the layer parameter string, which must be valid YAML
        layer_params = yaml.load(self.param_str_)
        self.object_number = layer_params['object_number']  # object_number =20
        if DEBUG:
            print 'object_number: {:}'.format(self.object_number)
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
        if DEBUG and self.iter_times % 200 == 0:
            print '+++++++++++++++++++++++++++++++++++++++++++++++++++++'
            print 'hyper_rois: {},{}'.format(len(hyper_rois), np.shape(hyper_rois))
            print 'hyper_cls_score:{},{},{}'.format(hyper_cls_score[0], np.shape(hyper_cls_score),
                                                    np.argmax(hyper_cls_score, axis=1))
            print 'hyper_labels:{}'.format(hyper_labels)
            print 'pred_labels:{}'.format(np.argmax(hyper_cls_score, axis=1))
            print 'hyper_pool:{}'.format(np.shape(hyper_pool))
            # print 'roi data labels:{},{}'.format(len(labels),np.shape(labels))
            print '+++++++++++++++++++++++++++++++++++++++++++++++++++++'
        per_image_classes = len(gt_boxes)
        hyper_pro_each = 64 / per_image_classes
        hyper_rois_ = hyper_rois[::hyper_pro_each + 1, 1:5] * 4

        self.iter_times += 1

        self.total_objects += len(hyper_rois)
        pred_labels = np.argmax(hyper_cls_score, axis=1)
        # print np.shape(np.equal(hyper_labels.astype(np.int16), pred_labels).nonzero())
        self.total_postives += np.shape(np.equal(hyper_labels.astype(np.int16), pred_labels).nonzero())[1]

        if self.iter_times % self.train_times == 0:
            print '~~~~~~~~~~~~~'
            print 'train_accuracy:{},{},{}'.format(self.total_postives, self.total_objects,
                                                   1.0 * self.total_postives / self.total_objects)
            print '~~~~~~~~~~~~~'
            if 1.0 * self.total_postives / self.total_objects >= 0.7:
                for index, gt_box in enumerate(gt_boxes):
                    hyper_cls_score_ = hyper_cls_score[index * hyper_pro_each:(index + 1) * hyper_pro_each + 1]
                    rois_ = hyper_rois[index * hyper_pro_each:(index + 1) * hyper_pro_each + 1, :]
                    label = int(gt_box[4]) - 1
                    gt_score = hyper_cls_score_[0, label]
                    keep = np.where(hyper_cls_score_[:, label] > gt_score)[0]
                    # print 'hyper_cls_score_:', len(hyper_cls_score_), hyper_cls_score_
                    # print 'hyper_rois_', len(rois_), rois_
                    # print 'gt_score:', gt_score
                    # print 'keep:', keep
                    if len(keep) > 0:
                        print '******************* find one   *****************'
                        print self.iter_times, "___________________"
                        hyper_rois_[index, :] = rois_[keep[0], 1:5] * 4
                        print hyper_rois_[index, :], rois_[keep[0], :]
            self.total_objects = 0
            self.total_postives = 0
        # print 'output_layer:',hyper_rois_
        hyper_labels_ = hyper_labels[::hyper_pro_each + 1].reshape((per_image_classes, 1))
        # print hyper_rois, hyper_labels
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
