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
import matplotlib.pyplot as plt
from hyper.hyper_proposal_beam_search import update_bbox_score

DEBUG = True

cache_path = './data/cache/'


class PsroiOutputLayer(caffe.Layer):
    """
    Outputs object detection proposals by applying estimated bounding-box
    transformations to a set of regular boxes (called "anchors").
    """
    iter_times = 0
    train_times = 1000  # each 1000 times to calculate train accuracy
    total_objects = 0
    total_positive = 0
    pse_gt_list = []

    def setup(self, bottom, top):
        # parse the layer parameter string, which must be valid YAML
        layer_params = yaml.load(self.param_str_)
        self.feat_stride = layer_params['feat_stride']  # feat_stride =4
        if DEBUG:
            print 'feat_stride: {:}'.format(self.feat_stride)
        top[0].reshape(1, 5)

    def forward(self, bottom, top):
        hyper_rois = bottom[0].data
        psroi_cls_score = bottom[1].data
        hyper_labels = bottom[2].data
        gt_boxes = bottom[3].data
        flipped = bottom[4].data
        img_index = bottom[5].data
        im_info = bottom[6].data[0]
        img_data = bottom[7].data
        simu_points = bottom[8].data
        psroipooled_cls_rois = bottom[9].data
        # print np.average(psroipooled_cls_rois[0,3,:])
        img_data_shape = np.shape(img_data)
        # print np.shape(img_data)
        img_data = img_data.reshape((img_data_shape[1], img_data_shape[2], img_data_shape[3]))
        # print np.shape(img_data)
        img_data = img_data.transpose((1, 2, 0))
        img_data += cfg.PIXEL_MEANS
        img_data = img_data[:, :, (2, 1, 0)]
        score_shape = np.shape(psroi_cls_score)
        hyper_rois = hyper_rois[:,:5]
        rois_shape = np.shape(hyper_rois)
        hyper_labels_shape = np.shape(hyper_labels)
        psroi_cls_score = psroi_cls_score.reshape((score_shape[0], score_shape[1]))
        hyper_rois = hyper_rois.reshape((rois_shape[0], rois_shape[1]))
        hyper_labels = hyper_labels.reshape((hyper_labels_shape[0],))
        one_epoc = 10022
        if DEBUG and self.iter_times % 200 == 0:
            print '+++++++++++++++++++++++++++++++++++++++++++++++++++++'
            print 'hyper_rois: {},{}'.format(len(hyper_rois), np.shape(hyper_rois))
            print 'hyer_labels:{},{}'.format(len(hyper_labels), np.shape(hyper_labels))
            print 'psroi_cls_score:{},{}'.format(np.shape(psroi_cls_score), psroi_cls_score[0])
            print 'hyper_labels:{}'.format(hyper_labels)
            print 'pred_labels:{}'.format(np.argmax(psroi_cls_score, axis=1))
            # print 'roi data labels:{},{}'.format(len(labels),np.shape(labels))
            print '+++++++++++++++++++++++++++++++++++++++++++++++++++++'
        per_image_classes = len(gt_boxes)
        fg_pro_each = 4  # 64 / per_image_classes + 1
        bg_pro_each = 0  # 64 / per_image_classes
        hyper_rois_ = hyper_rois[::fg_pro_each + bg_pro_each, 1:5] * self.feat_stride
        hyper_labels_ = hyper_labels[::fg_pro_each + bg_pro_each].reshape((len(gt_boxes), 1))
        self.iter_times += 1
        self.total_objects += len(hyper_rois)
        pred_labels = np.argmax(psroi_cls_score, axis=1)
        self.total_positive += np.shape(np.equal(hyper_labels.astype(np.int16), pred_labels).nonzero())[1]
        train_accuracy = 1.0 * self.total_positive / self.total_objects
        # feature_shape = np.shape(hyper_features)
        # hyper_features = hyper_features.reshape((feature_shape[1], feature_shape[2], feature_shape[3]))
        # print 'self.iter_times:{},train_accuracy:{}'.format(self.iter_times,train_accuracy)
        if self.iter_times % self.train_times == 0:
            print '~~~~~~~~~~~~~'
            print 'train_accuracy:{},{},{}'.format(self.total_positive, self.total_objects, train_accuracy)
            print '~~~~~~~~~~~~~'
            self.total_objects = 0
            self.total_positive = 0

        # print 'psroi_gt_boxes:', gt_boxes
        if train_accuracy > 0. and self.iter_times > 0:
            bbox_scores = []

            fig2, ax2 = plt.subplots(figsize=(12, 12))
            show_box = ax2.get_position()
            ax2.set_position([show_box.x0,show_box.y0,show_box.width*0.8,show_box.height])
            ax2.imshow(img_data.astype(np.uint8), aspect='equal')

            for index, gt_box in enumerate(gt_boxes):
                simu_point = simu_points[index]
                hyper_cls_score_ = psroi_cls_score[
                                   index * (fg_pro_each + bg_pro_each):(index + 1) * (fg_pro_each + bg_pro_each)]
                rois_ = hyper_rois[index * (fg_pro_each + bg_pro_each):(index + 1) * (fg_pro_each + bg_pro_each), :]
                pred_label_ = pred_labels[index * (fg_pro_each + bg_pro_each):(index + 1) * (fg_pro_each + bg_pro_each)]
                hyper_cls_score_ = hyper_cls_score_[0:fg_pro_each]
                rois_ = rois_[0:fg_pro_each]
                pred_label_ = pred_label_[0:fg_pro_each]
                label = int(gt_box[4])
                # score_bbox_center = (np.sqrt(((rois_[:, 0] + rois_[:, 2]) / 2.0 - simu_point[0]) ** 2 +
                #                      ((rois_[:, 1] + rois_[:, 3]) / 2.0 - simu_point[1]) ** 2)) / 2000
                # #
                # print 'hyper_cls_score:',hyper_cls_score_[:, label],np.exp(-score_bbox_center)
                # hyper_cls_score_[:, label] = np.multiply(hyper_cls_score_[:, label],
                #                                          np.exp(-score_bbox_center))
                # print 'score_bbox_center',np.shape(score_bbox_center), score_bbox_center,
                # print 'hyper_cls_score:', hyper_cls_score_[:, label]
                # print 'background_score:',hyper_cls_score_[:,0]
                # print 'pred_label_:',pred_label_
                # print 'psroi_gt_boxes:',gt_boxes
                bbox_scores.append(hyper_cls_score_[:, label])

                # keep = np.where((hyper_cls_score_[:, label] >= 0) & (pred_label_ == label))[0]
                keep = np.where((hyper_cls_score_[:, label] >= 0))[0]
                if len(keep) >= 1:
                    rois_ = rois_[keep, :]
                    hyper_cls_score_ = hyper_cls_score_[keep, :]

                    # order = hyper_cls_score_[:, label].argsort()[::-1]
                    # rois_ = rois_[order, :]
                    # hyper_cls_score_ = hyper_cls_score_[order, :]

                    best_score = np.max(hyper_cls_score_[:, label])
                    current_best_score = best_score
                    current_best_roi = rois_[np.argmax(hyper_cls_score_[:,label])]

                    for i, rois_i in enumerate(rois_):
                        # if highest score bounding boxes contained another one, don't consider this one
                        # if (best_roi[1] <= rois_i[1] and best_roi[2] <= rois_i[2]
                        #     and best_roi[3] >= rois_i[3] and best_roi[4] >= rois_i[4]):
                        if np.abs(best_score / hyper_cls_score_[i, label]) <= 2:
                            best_score = hyper_cls_score_[i, label]
                            best_roi = rois_i

                            roi = best_roi[1:5]
                            roi = roi * self.feat_stride
                            # plt.colorbar(gci, fraction=0.046, pad=0.04)
                            ax2.add_patch(

                                plt.Rectangle((roi[0], roi[1]),
                                              roi[2] - roi[0],
                                              roi[3] - roi[1], fill=False,
                                              edgecolor=np.random.rand(3, 1).reshape(3,),
                                              linewidth=1.5,label= ('0>' if i % 2 == 1 else '') +
                                              'best_roi:'+str(best_score)+',area:'+str((roi[2] - roi[0])*(roi[3] - roi[1])))
                            )

                    current_best_roi = current_best_roi[1:5] * self.feat_stride
                    ax2.add_patch(

                        plt.Rectangle((current_best_roi[0], current_best_roi[1]),
                                      current_best_roi[2] - current_best_roi[0],
                                      current_best_roi[3] - current_best_roi[1], fill=False,
                                      edgecolor='yellow', linewidth=1, label='current_best:'+str(current_best_score))
                    )
                    hyper_rois_[index, :] = current_best_roi
                    if self.iter_times % 200 == 0:
                        print '****************best roi***********'
                        print 'gt_score:', current_best_score, 'gt_roi:', current_best_roi
                        print 'rois_:', rois_
                        print 'hyper_score:', hyper_cls_score_[:, label]
                        # print 'best_roi:', best_roi, 'best_score:', best_score
                        print '***********************************'
                elif len(rois_) == 1:
                    hyper_rois_[index, :] = rois_[keep[0], 1:5] * self.feat_stride
            update_bbox_score(bbox_scores)
            plt.legend(loc='best', fontsize=10,bbox_to_anchor=(1.0,1.0),borderaxespad = 0.)
            plt.show()
        if self.iter_times % 8 == 0:  # 8 iterations for one image
            if len(self.pse_gt_list) < one_epoc:
                self.pse_gt_list.append({'img_index': int(img_index),
                                         'flipped': int(flipped),
                                         'rpn_simu_gt_boxes': hyper_rois_ / im_info[2]})
        if len(self.pse_gt_list) == one_epoc:
            if self.iter_times == 8*one_epoc:
                self.pse_gt_list.sort(key=take_key)
            elif self.iter_times > 8*one_epoc:
                save_pse_results(self, int(img_index), int(flipped), hyper_rois_ / im_info[2], one_epoc,
                                 'classifi_bg_valid', self.iter_times)
        hyper_gt_boxes = np.concatenate((hyper_rois_, hyper_labels_), axis=1)
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
