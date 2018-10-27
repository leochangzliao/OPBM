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
from fast_rcnn.nms_wrapper import nms

import os
import cPickle
import logging
import matplotlib.pyplot as plt

DEBUG = False


class HyperFeatureExtract(caffe.Layer):
    """
    Outputs object detection proposals by applying estimated bounding-box
    transformations to a set of regular boxes (called "anchors").
    """

    def setup(self, bottom, top):
        # parse the layer parameter string, which must be valid YAML
        layer_params = yaml.load(self.param_str_)
        if DEBUG:
            pass
            # rois blob: holds R regions of interest, each is a 5-tuple
            # (n, x1, y1, x2, y2) specifying an image batch index n and a
            # rectangle (x1, y1, x2, y2)
            # top[0].reshape(1, 3)
        top[0].reshape(1, np.shape(bottom[0].data)[1] * 3, np.shape(bottom[0].data)[2], np.shape(bottom[0].data)[3])
        self.current = 0

    def forward(self, bottom, top):
        hyper_feature_map1 = bottom[0].data
        hyper_feature_map2 = bottom[1].data
        hyper_feature_map3 = bottom[2].data
        conv_features = [bottom[i].data for i in xrange(3, len(bottom))]
        if DEBUG:
            print '+++++++++++++++++++++++++++++++++++++++++++++++++++++'
            print 'hyper_feature_map1: ({})'.format(np.shape(hyper_feature_map1))
            print 'hyper_feature_map2: {}'.format(np.shape(hyper_feature_map2))
            print 'hyper_feature_map3:{}'.format(np.shape(hyper_feature_map3))
            print '+++++++++++++++++++++++++++++++++++++++++++++++++++++'
        if np.shape(hyper_feature_map1)[2] is not np.shape(hyper_feature_map3)[2]:
            shape_diff = np.shape(hyper_feature_map3)[2] - np.shape(hyper_feature_map1)[2]
            if shape_diff == 1:
                hyper_feature_map3 = np.delete(hyper_feature_map3, [0], axis=2)
            elif shape_diff == 2:
                hyper_feature_map3 = np.delete(hyper_feature_map3, [0, np.shape(hyper_feature_map3)[2] - 1], axis=2)
            else:
                hyper_feature_map3 = np.delete(hyper_feature_map3,
                                               [0, np.shape(hyper_feature_map3)[2] - 1,
                                                np.shape(hyper_feature_map3)[2] - 2], axis=2)
        if np.shape(hyper_feature_map1)[3] is not np.shape(hyper_feature_map3)[3]:
            shape_diff = np.shape(hyper_feature_map3)[3] - np.shape(hyper_feature_map1)[3]
            if shape_diff == 1:
                hyper_feature_map3 = np.delete(hyper_feature_map3, [0], axis=3)
            elif shape_diff == 2:
                hyper_feature_map3 = np.delete(hyper_feature_map3, [0, np.shape(hyper_feature_map3)[3] - 1], axis=3)
            else:
                hyper_feature_map3 = np.delete(hyper_feature_map3,
                                               [0, np.shape(hyper_feature_map3)[3] - 1,
                                                np.shape(hyper_feature_map3)[3] - 2], axis=3)
        assert np.shape(hyper_feature_map1) == np.shape(hyper_feature_map3)
        assert np.shape(hyper_feature_map2) == np.shape(hyper_feature_map3)
        # vis_feature(self, hyper_feature_map1, hyper_feature_map2, hyper_feature_map3, conv_features)
        # min_ =
        blob = np.concatenate((hyper_feature_map1, hyper_feature_map2, hyper_feature_map3), axis=1)
        # print blob.shape, (1, 126, np.shape(hyper_feature_map1)[2], np.shape(hyper_feature_map1)[3])
        assert blob.shape == (1, np.shape(hyper_feature_map1)[1] + np.shape(hyper_feature_map2)[1] +
                              np.shape(hyper_feature_map3)[1],
                              np.shape(hyper_feature_map1)[2], np.shape(hyper_feature_map1)[3])
        top[0].reshape(*(blob.shape))
        top[0].data[...] = blob

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


def norm(x, s=1.0):
    print x.min(), x.max()
    x -= x.min()
    x /= x.max()
    return x * s


def vis_feature(self, data1, data2, data3, conv_features):
    img_data_shape = np.shape(data1)
    # print np.shape(img_data)
    img_data1 = data1.reshape((img_data_shape[1], img_data_shape[2], img_data_shape[3]))
    img_data2 = data2.reshape((img_data_shape[1], img_data_shape[2], img_data_shape[3]))
    img_data3 = data3.reshape((img_data_shape[1], img_data_shape[2], img_data_shape[3]))

    # img_data = img_data.transpose((1, 2, 0))
    # for index in xrange(img_data_shape[1]):
    #     # img_data1 = np.average(img_data1,axis=0)
    #     # img_data2 = np.average(img_data2, axis=0)
    #     # img_data3 = np.average(img_data3, axis=0)
    #     img_data1_ = img_data1[index]
    #     img_data2_ = img_data2[index]
    #     img_data3_ = img_data3[index]
    #     plt.figure(1)
    #     plt.subplot(141)
    #     gci = plt.imshow(norm(img_data1_, 255.0))
    #     plt.colorbar(gci,fraction=0.046, pad=0.04)
    #     plt.subplot(142)
    #     gci = plt.imshow(norm(img_data2_, 255.0))
    #     plt.colorbar(gci,fraction=0.046, pad=0.04)
    #     plt.subplot(143)
    #     gci = plt.imshow(norm(img_data3_, 255.0))
    #     plt.colorbar(gci,fraction=0.046, pad=0.04)
    #     plt.subplot(144)
    #     gci = plt.imshow(norm(img_data3_+img_data2_+img_data1_, 255.0))
    #     plt.colorbar(gci, fraction=0.046, pad=0.04)
    #     plt.show()
    plt.figure(1)
    print 'hyper_conv1'
    gci = plt.imshow(norm(np.average(img_data1, axis=0), 255.0))
    plt.colorbar(gci, fraction=0.046, pad=0.04)
    plt.title('hyper_conv1')

    plt.figure(2)
    print 'hyper_conv3'
    gci = plt.imshow(norm(np.average(img_data2, axis=0), 255.0))
    plt.colorbar(gci, fraction=0.046, pad=0.04)
    plt.title('hyper_conv3')

    plt.figure(3)
    print 'hyper_deconv5'
    gci = plt.imshow(norm(np.average(img_data3, axis=0), 255.0))
    plt.colorbar(gci, fraction=0.046, pad=0.04)
    plt.title('hyper_deconv5')

    plt.figure(4)
    gci = plt.imshow(norm(np.average(norm(img_data3, 255.0), axis=0) + np.average(norm(img_data2, 255.0), axis=0)
                          + np.average(norm(img_data1, 255.0), axis=0), 255.0))
    plt.colorbar(gci, fraction=0.046, pad=0.04)
    print 'hyper_merged'
    plt.title('hyper_merged')

    figure_num = 4
    category = ['bird', 'car', 'person']

    for features in conv_features:
        figure_num += 1
        shape = np.shape(features)
        features = features.reshape((shape[1], shape[2], shape[3]))
        plt.figure(figure_num)
        print 'conv' + str(figure_num - 4)
        gci = plt.imshow(norm(np.average(features, axis=0), 255.0))
        plt.colorbar(gci, fraction=0.046, pad=0.04)
        plt.title('conv' + str(figure_num - 4))
        # plt.savefig('/home/leochang/test/' + str(category[self.current]) + '/conv_' + str(figure_num - 4))
    self.current += 1
    plt.show()
