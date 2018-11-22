# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

import os
from datasets.imdb import imdb
import datasets.ds_utils as ds_utils
import xml.etree.ElementTree as ET
import numpy as np
import scipy.sparse
import scipy.io as sio
import utils.cython_bbox
from utils.cython_bbox_grid import bbox_overlaps_grid
from utils.cython_bbox import bbox_overlaps
import cPickle
import subprocess
import uuid
from voc_eval import voc_eval
from fast_rcnn.config import cfg
import cv2
import PIL
from utils.timer import Timer
import glob
from utils.cython_bbox import bbox_overlaps
import warnings

warnings.filterwarnings('error')


class pascal_voc(imdb):
    def __init__(self, image_set, year, devkit_path=None):
        imdb.__init__(self, 'voc_' + year + '_' + image_set)
        self._year = year
        self._image_set = image_set
        self._devkit_path = self._get_default_path() if devkit_path is None \
            else devkit_path
        self._data_path = os.path.join(self._devkit_path, 'VOC' + self._year)
        self._classes = ('__background__',  # always index 0
                         'aeroplane', 'bicycle', 'bird', 'boat',
                         'bottle', 'bus', 'car', 'cat', 'chair',
                         'cow', 'diningtable', 'dog', 'horse',
                         'motorbike', 'person', 'pottedplant',
                         'sheep', 'sofa', 'train', 'tvmonitor')
        self._class_to_ind = dict(zip(self.classes, xrange(self.num_classes)))
        self._image_ext = '.jpg'
        self._image_index = self._load_image_set_index()
        # Default to roidb handler
        # self._roidb_handler = self.selective_search_roidb
        self._salt = str(uuid.uuid4())
        self._comp_id = 'comp4'

        # PASCAL specific config options
        self.config = {'cleanup': True,
                       'use_salt': True,
                       'use_diff': False,
                       'matlab_eval': False,
                       'rpn_file': None,
                       'min_size': 2}

        assert os.path.exists(self._devkit_path), \
            'VOCdevkit path does not exist: {}'.format(self._devkit_path)
        assert os.path.exists(self._data_path), \
            'Path does not exist: {}'.format(self._data_path)

    def image_path_at(self, i):
        """
        Return the absolute path to image i in the image sequence.
        """
        return self.image_path_from_index(self._image_index[i])

    def image_path_from_index(self, index):
        """
        Construct an image path from the image's "index" identifier.
        """
        image_path = os.path.join(self._data_path, 'JPEGImages',
                                  index + self._image_ext)
        assert os.path.exists(image_path), \
            'Path does not exist: {}'.format(image_path)
        return image_path

    def _load_image_set_index(self):
        """
        Load the indexes listed in this dataset's image set file.
        """
        # Example path to image set file:
        # self._devkit_path + /VOCdevkit2007/VOC2007/ImageSets/Main/val.txt
        image_set_file = os.path.join(self._data_path, 'ImageSets', 'Main',
                                      self._image_set + '.txt')
        assert os.path.exists(image_set_file), \
            'Path does not exist: {}'.format(image_set_file)
        with open(image_set_file) as f:
            image_index = [x.strip() for x in f.readlines()]
        return image_index

    def _get_default_path(self):
        """
        Return the default path where PASCAL VOC is expected to be installed.
        """
        return os.path.join(cfg.DATA_DIR, 'VOCdevkit' + self._year)

    def gt_roidb(self):
        """
        Return the database of ground-truth regions of interest.

        This function loads/saves from/to a cache file to speed up future calls.
        """
        cache_file = os.path.join(self.cache_path, self.name + '_gt_roidb_with_simu_points.pkl')
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                roidb = cPickle.load(fid)
            print '{} gt roidb loaded from {}'.format(self.name, cache_file)
            return roidb

        gt_roidb = [self._load_pascal_annotation(index)
                    for index in self.image_index]
        with open(cache_file, 'wb') as fid:
            cPickle.dump(gt_roidb, fid, cPickle.HIGHEST_PROTOCOL)
        print 'wrote gt roidb to {}'.format(cache_file)

        return gt_roidb

    def gt_roidb_with_simu_points(self):
        """
        Return the database of ground-truth regions of interest.

        This function loads/saves from/to a cache file to speed up future calls.
        """
        cache_file = os.path.join(self.cache_path, self.name + '_gt_roidb_with_simu_points.pkl')
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                roidb = cPickle.load(fid)
            print '{} gt roidb loaded from {}'.format(self.name, cache_file)
            return roidb

        gt_roidb = [self._load_pascal_annotation(index)
                    for index in self.image_index]
        with open(cache_file, 'wb') as fid:
            cPickle.dump(gt_roidb, fid, cPickle.HIGHEST_PROTOCOL)
        print 'wrote gt roidb to {}'.format(cache_file)

        return gt_roidb

    def selective_search_roidb(self):
        """
        Return the database of selective search regions of interest.
        Ground-truth ROIs are also included.

        This function loads/saves from/to a cache file to speed up future calls.
        """
        cache_file = os.path.join(self.cache_path,
                                  self.name + '_selective_search_roidb.pkl')

        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                roidb = cPickle.load(fid)
            print '{} ss roidb loaded from {}'.format(self.name, cache_file)
            return roidb

        if int(self._year) == 2007 or self._image_set != 'test':
            gt_roidb = self.gt_roidb()
            ss_roidb = self._load_selective_search_roidb(gt_roidb)
            roidb = imdb.merge_roidbs(gt_roidb, ss_roidb)
        else:
            roidb = self._load_selective_search_roidb(None)
        with open(cache_file, 'wb') as fid:
            cPickle.dump(roidb, fid, cPickle.HIGHEST_PROTOCOL)
        print 'wrote ss roidb to {}'.format(cache_file)

        return roidb

    def rpn_roidb(self):
        if int(self._year) == 2007 or self._image_set != 'test':
            gt_roidb = self.gt_roidb()
            rpn_roidb = self._load_rpn_roidb(gt_roidb)
            roidb = imdb.merge_roidbs(gt_roidb, rpn_roidb)
        else:
            roidb = self._load_rpn_roidb(None)

        return roidb

    def _load_rpn_roidb(self, gt_roidb):
        filename = self.config['rpn_file']
        print 'loading {}'.format(filename)
        assert os.path.exists(filename), \
            'rpn data not found at: {}'.format(filename)
        with open(filename, 'rb') as f:
            box_list = cPickle.load(f)
        return self.create_roidb_from_box_list(box_list, gt_roidb)

    def _load_selective_search_roidb(self, gt_roidb):
        filename = os.path.abspath(os.path.join(cfg.DATA_DIR,
                                                'selective_search_data',
                                                self.name + '.mat'))
        assert os.path.exists(filename), \
            'Selective search data not found at: {}'.format(filename)
        raw_data = sio.loadmat(filename)['boxes'].ravel()

        box_list = []
        for i in xrange(raw_data.shape[0]):
            boxes = raw_data[i][:, (1, 0, 3, 2)] - 1
            keep = ds_utils.unique_boxes(boxes)
            boxes = boxes[keep, :]
            keep = ds_utils.filter_small_boxes(boxes, self.config['min_size'])
            boxes = boxes[keep, :]
            box_list.append(boxes)

        return self.create_roidb_from_box_list(box_list, gt_roidb)

    def _load_pascal_annotation(self, index):
        """
        Load image and bounding boxes info from XML file in the PASCAL VOC
        format.
        """
        filename = os.path.join(self._data_path, 'Annotations', index + '.xml')
        tree = ET.parse(filename)
        objs = tree.findall('object')
        if self.config['use_diff']:
            # Exclude the samples labeled as difficult
            non_diff_objs = [
                obj for obj in objs if int(obj.find('difficult').text) == 0]
            if len(non_diff_objs) != len(objs):
                print 'Removed {} difficult objects'.format(
                    len(objs) - len(non_diff_objs))
            objs = non_diff_objs
        num_objs = len(objs)

        boxes = np.zeros((num_objs, 4), dtype=np.uint16)
        gt_points = np.zeros((num_objs, 2), dtype=np.uint16)
        simu_points = np.zeros((num_objs, 3), dtype=np.uint16)
        gt_classes = np.zeros((num_objs), dtype=np.int32)
        overlaps = np.zeros((num_objs, self.num_classes), dtype=np.float32)
        # "Seg" area for pascal is just the box area
        seg_areas = np.zeros((num_objs), dtype=np.float32)

        img_size = tree.find('size')
        width = float(img_size.find('width').text)
        height = float(img_size.find('height').text)
        print 'height:', height, 'width:', width
        # Load object bounding boxes into a data frame.
        for ix, obj in enumerate(objs):
            bbox = obj.find('bndbox')
            # Make pixel indexes 0-based
            x1 = float(bbox.find('xmin').text) - 1
            y1 = float(bbox.find('ymin').text) - 1
            x2 = float(bbox.find('xmax').text) - 1
            y2 = float(bbox.find('ymax').text) - 1
            cls = self._class_to_ind[obj.find('name').text.lower().strip()]
            boxes[ix, :] = [x1, y1, x2, y2]
            gt_point_x = (x1 + x2) / 2
            gt_point_y = (y1 + y2) / 2
            gt_points[ix, :] = [gt_point_x, gt_point_y]
            is_small_object = False
            small_object_thresh = 7  # 7 pixels to center
            big_object_thresh = 20  # 20 pixels to center

            if (x2 - x1) * (y2 - y1) / (width * height) < 0.1:
                is_small_object = True
                low_x = 0 if int(gt_point_x - small_object_thresh) < 0 else int(gt_point_x - small_object_thresh)
                high_x = width if int(gt_point_x + small_object_thresh) > width else int(
                    gt_point_x + small_object_thresh)
                low_y = 0 if int(gt_point_y - small_object_thresh) < 0 else int(gt_point_y - small_object_thresh)
                high_y = height if int(gt_point_y + small_object_thresh) > height else int(
                    gt_point_y + small_object_thresh)
            else:
                low_x = 0 if int(gt_point_x - big_object_thresh) < 0 else int(gt_point_x - big_object_thresh)
                high_x = width if int(gt_point_x + big_object_thresh) > width else int(
                    gt_point_x + big_object_thresh)
                low_y = 0 if int(gt_point_y - big_object_thresh) < 0 else int(gt_point_y - big_object_thresh)
                high_y = height if int(gt_point_y + big_object_thresh) > height else int(
                    gt_point_y + big_object_thresh)
            simu_x = np.random.randint(low=low_x, high=high_x)
            simu_y = np.random.randint(low=low_y, high=high_y)

            gt_classes[ix] = cls
            overlaps[ix, cls] = 1.0
            seg_areas[ix] = (x2 - x1 + 1) * (y2 - y1 + 1)
            simu_points[ix, :] = [simu_x, simu_y, int(is_small_object)]
        overlaps = scipy.sparse.csr_matrix(overlaps)

        return {'boxes': boxes,
                'gt_points': gt_points,
                'simu_points': simu_points,
                'gt_classes': gt_classes,
                }


def load_selective_search_roidb(dataset_name):
    name = dataset_name
    filename = os.path.abspath(os.path.join(cfg.DATA_DIR,
                                            'selective_search_data',
                                            name + '.mat'))
    assert os.path.exists(filename), \
        'Selective search data not found at: {}'.format(filename)
    raw_data = sio.loadmat(filename)['boxes'].ravel()

    box_list = []
    for i in xrange(raw_data.shape[0]):
        boxes = raw_data[i][:, (1, 0, 3, 2)] - 1
        keep = ds_utils.unique_boxes(boxes)
        boxes = boxes[keep, :]
        keep = ds_utils.filter_small_boxes(boxes, 2)
        boxes = boxes[keep, :]
        box_list.append(boxes)
    return box_list


def gt_roidb_with_simu_points(cache_path, dataset_name):
    """
    Return the database of ground-truth regions of interest.

    This function loads/saves from/to a cache file to speed up future calls.
    """
    cache_file = os.path.join(cache_path, dataset_name + '_gt_roidb_with_simu_points.pkl')
    if os.path.exists(cache_file):
        with open(cache_file, 'rb') as fid:
            roidb = cPickle.load(fid)
        print '{} gt roidb loaded from {}'.format(dataset_name, cache_file)
        return roidb


def average_bounding_boxes(d, roidb, boxlist):
    import matplotlib.pyplot as plt

    print len(boxlist), len(roidb)
    simu_gt_boxes_list = []
    for i_roi in xrange(len(roidb)):
        im = cv2.imread(d.image_path_at(i_roi))
        print 'image_index:', i_roi
        im = im[:, :, (2, 1, 0)]
        gt_boxes = roidb[i_roi]['boxes']
        gt_points = roidb[i_roi]['gt_points']
        simu_points = roidb[i_roi]['simu_points']
        simu_gt_boxes = np.zeros((len(gt_points), 4))
        for i_gt in xrange(len(gt_boxes)):
            bbox = gt_boxes[i_gt, :4]
            gt_p = gt_points[i_gt, :2]
            plt.cla()
            plt.imshow(im)
            plt.gca().add_patch(
                plt.Rectangle((bbox[0], bbox[1]),
                              bbox[2] - bbox[0],
                              bbox[3] - bbox[1], fill=False,
                              edgecolor='r', linewidth=3),
            )
            plt.gca().add_patch(
                plt.Rectangle((gt_p[0], gt_p[1]),
                              2,
                              2, fill=True,
                              edgecolor='r', linewidth=3)
            )
            plt.gca().add_patch(
                plt.Rectangle((simu_points[i_gt, 0], simu_points[i_gt, 1]),
                              2,
                              2, fill=True,
                              edgecolor='g', linewidth=3)
            )
            plt.title('{}  {}'.format('small object?', str(simu_points[i_gt, 2])))

            # for box in boxlist[i]:
            #     plt.gca().add_patch(
            #         plt.Rectangle((box[0], box[1]),
            #                       box[2] - box[0],
            #                       box[3] - box[1], fill=False,
            #                       edgecolor='blue', linewidth=1),
            #     )
            # print 'orginal len(boxlist[i])',len(boxlist[i]),np.dtype(simu_points[i_gt,2]),simu_points[i_gt,2],im.shape

            filted_boxlist = filter_selective_boxlist_average(im, simu_points, i_gt, boxlist, i_roi)
            # print 'len(filted_boxlist[i])', len(filted_boxlist)
            box_array = np.array(filted_boxlist)
            box_mean = np.mean(box_array, axis=0)
            plt.gca().add_patch(
                plt.Rectangle((box_mean[0], box_mean[1]),
                              box_mean[2] - box_mean[0],
                              box_mean[3] - box_mean[1], fill=False,
                              edgecolor='yellow', linewidth=1),
            )
            plt.show()
            simu_gt_boxes[i_gt, :] = box_mean
        simu_gt_boxes_list.append(simu_gt_boxes)
    cache_file = os.path.join(d.cache_path, d.name + '_gt_roidb_with_simu_bbox.pkl')
    with open(cache_file, 'wb') as fid:
        cPickle.dump(simu_gt_boxes_list, fid, cPickle.HIGHEST_PROTOCOL)
    print 'wrote gt roidb to {}'.format(cache_file)


def filter_selective_boxlist_average(im, simu_points, i_gt, boxlist, i):
    margin_thresh = 0
    filted_boxlist = []
    bbox_size_thresh = 1
    if simu_points[i_gt, 2] == 1:
        bbox_size_thresh = 0.15
    while len(filted_boxlist) == 0:
        for box in boxlist[i]:
            if (box[0] - margin_thresh < simu_points[i_gt, 0]) and (box[2] + margin_thresh > simu_points[i_gt, 0]) \
                    and (box[1] - margin_thresh < simu_points[i_gt, 1]) and (
                            box[3] + margin_thresh > simu_points[i_gt, 1]):

                if 1.0 * (box[2] - box[0]) * (box[3] - box[1]) / (
                                1.0 * im.shape[0] * im.shape[1]) < bbox_size_thresh:
                    # print 1.0*(bbox[2]-bbox[0])*(bbox[3]-bbox[1])/(1.0*im.shape[0]*im.shape[1])
                    # plt.gca().add_patch(
                    #     plt.Rectangle((box[0], box[1]),
                    #                   box[2] - box[0],
                    #                   box[3] - box[1], fill=False,
                    #                   edgecolor='blue', linewidth=1),
                    # )
                    if bbox_size_thresh == 1:  # if gt_box is big, just keep boxes whose size more than 0.05
                        if 1.0 * (box[2] - box[0]) * (box[3] - box[1]) / (
                                        1.0 * im.shape[0] * im.shape[1]) > 0.05:
                            filted_boxlist.append(box)
                    else:  # if gt_box is small, just keep boxes whose size less than 0.15
                        filted_boxlist.append(box)
        bbox_size_thresh = bbox_size_thresh + 0.05
        margin_thresh = margin_thresh + 1
        # print bbox_size_thresh,margin_thresh
    return filted_boxlist


def filter_selective_boxlist_heatmap(im, simu_points, i_gt, boxlist, i):
    filted_boxlist = []
    Top_N_Boxes_small = 8
    Top_N_Boxes_big = 8
    aspetct_ratio1 = 0.5
    aspetct_ratio2 = 1
    aspetct_ratio3 = 2
    area_ratio_small = 1
    area_ratio_big_low = 0.
    area_ratio_big_high = 1.
    print area_ratio_big_low, area_ratio_small, Top_N_Boxes_small, Top_N_Boxes_big, area_ratio_big_high

    def take_elem(elem):
        return elem['sum_diff']

    for box in boxlist[i]:
        box = box.astype(np.float)
        margin_thresh = 5  # for robustness to add some pixels
        if (box[0] - margin_thresh < simu_points[i_gt, 0]) and (box[2] + margin_thresh > simu_points[i_gt, 0]) \
                and (box[1] - margin_thresh < simu_points[i_gt, 1]) and (
                        box[3] + margin_thresh > simu_points[i_gt, 1]):
            distance_x = [np.abs(simu_points[i_gt, 0] - box[0]) + 0.001, np.abs(box[2] - simu_points[i_gt, 0]) + 0.001]
            distance_y = [np.abs(simu_points[i_gt, 1] - box[1] + 0.001), np.abs(box[3] - simu_points[i_gt, 1]) + 0.001]
            # print np.max(distance_x),np.min(distance_x)
            diff_x = np.abs(np.max(distance_x) / np.min(distance_x))
            diff_y = np.abs(np.max(distance_y) / np.min(distance_y))
            ratio = np.abs((box[2] - box[0]) / (box[3] - box[1]))
            sum_diff = (diff_x + diff_y)
            #
            #        * np.exp(
            # -np.linalg.norm([0.5 * (box[0] + box[2]) - 1.0 * simu_points[i_gt, 0],
            #                  0.5 * (box[1] + box[3]) - 1.0 * simu_points[i_gt, 1]])/200)
            object_area = (box[2] - box[0]) * (box[3] - box[1]) / (im.shape[0] * im.shape[1])
            if int(simu_points[i_gt, 2]) == 1:  # small object

                if object_area <= area_ratio_small:
                    filted_boxlist.append({'sum_diff': sum_diff,
                                           'boxes': box,
                                           'ratio': ratio})
            else:
                if object_area >= area_ratio_big_low and object_area <= area_ratio_big_high:
                    filted_boxlist.append({'sum_diff': sum_diff,
                                           'boxes': box,
                                           'ratio': ratio})
    filted_boxlist.sort(key=take_elem)
    # Top_N_Boxes = len(filted_boxlist)
    # if len(filted_boxlist) >= Top_N_Boxes:
    #     filterd = filted_boxlist[:Top_N_Boxes]
    # else:
    #     filterd = filted_boxlist
    # ratiolist = np.array([b['ratio'] for b in filterd])
    # aspetct_nums = [len(np.where(ratiolist <= aspetct_ratio1)[0]), len(np.where(ratiolist >= aspetct_ratio3)[0]),
    #                 len(np.where((ratiolist < aspetct_ratio3) & (ratiolist > aspetct_ratio1))[0])]
    # max_ratio = np.argmax(aspetct_nums)
    # min_ratio = np.argmin(aspetct_nums)
    # boxes_low_aspect = np.array([b['boxes'] for b in filterd])[np.where(ratiolist <= aspetct_ratio1)[0], :]
    # boxes_high_aspect = np.array([b['boxes'] for b in filterd])[np.where(ratiolist >= aspetct_ratio3)[0], :]
    # boxes_normal_aspect = np.array([b['boxes'] for b in filterd])[
    #                       np.where((ratiolist < aspetct_ratio3) & (ratiolist > aspetct_ratio1))[0], :]
    #
    # if max_ratio == 0 and min_ratio == 1:
    #     low, high, normal = 0.5, 0.3, 0.2
    #     if len(boxes_high_aspect) == 0 and len(boxes_normal_aspect) == 0:
    #         low, high, normal = 1.0, 0., 0.
    #     elif len(boxes_high_aspect) == 0:
    #         low, high, normal = 0.65, 0., 0.35
    #     elif len(boxes_normal_aspect) == 0:
    #         low, high, normal = 0.6, 0.4, 0.
    #
    # elif max_ratio == 0 and min_ratio == 2:
    #     low, high, normal = 0.5, 0.2, 0.3
    #     if len(boxes_high_aspect) == 0 and len(boxes_normal_aspect) == 0:
    #         low, high, normal = 1.0, 0., 0.
    #     elif len(boxes_high_aspect) == 0:
    #         low, high, normal = 0.6, 0., 0.4
    #     elif len(boxes_normal_aspect) == 0:
    #         low, high, normal = 0.65, 0.35, 0.
    #
    # elif max_ratio == 1 and min_ratio == 0:
    #     low, high, normal = 0.3, 0.5, 0.2
    #     if len(boxes_low_aspect) == 0 and len(boxes_normal_aspect) == 0:
    #         low, high, normal = 0., 1., 0.
    #     elif len(boxes_low_aspect) == 0:
    #         low, high, normal = 0., 0.65, 0.35
    #     elif len(boxes_normal_aspect) == 0:
    #         low, high, normal = 0.4, 0.6, 0.
    #
    # elif max_ratio == 1 and min_ratio == 2:
    #     low, high, normal = 0.2, 0.5, 0.3
    #     if len(boxes_low_aspect) == 0 and len(boxes_normal_aspect) == 0:
    #         low, high, normal = 0., 1., 0.
    #     elif len(boxes_low_aspect) == 0:
    #         low, high, normal = 0., 0.6, 0.4
    #     elif len(boxes_normal_aspect) == 0:
    #         low, high, normal = 0.35, 0.65, 0.
    #
    # elif max_ratio == 2 and min_ratio == 0:
    #     low, high, normal = 0.3, 0.2, 0.5
    #     if len(boxes_low_aspect) == 0 and len(boxes_high_aspect) == 0:
    #         low, high, normal = 0., 0., 1.
    #     elif len(boxes_low_aspect) == 0:
    #         low, high, normal = 0., 0.35, 0.65
    #     elif len(boxes_high_aspect) == 0:
    #         low, high, normal = 0.4, 0., 0.6
    #
    # else:
    #     low, high, normal = 0.2, 0.3, 0.5
    #     if len(boxes_low_aspect) == 0 and len(boxes_high_aspect) == 0:
    #         low, high, normal = 0., 0., 1.
    #     elif len(boxes_low_aspect) == 0:
    #         low, high, normal = 0., 0.4, 0.6
    #     elif len(boxes_high_aspect) == 0:
    #         low, high, normal = 0.35, 0., 0.65
    # if len(boxes_low_aspect) == 0:
    #     boxes_low_aspect = np.array([0, 0, 0, 0])
    # if len(boxes_high_aspect) == 0:
    #     boxes_high_aspect = np.array([0, 0, 0, 0])
    # if len(boxes_normal_aspect) == 0:
    #     boxes_normal_aspect = np.array([0, 0, 0, 0])
    #
    # assert len(boxes_low_aspect) is not 0
    # assert len(boxes_high_aspect) is not 0
    # assert len(boxes_normal_aspect) is not 0
    #
    # return np.average(boxes_low_aspect, axis=0) * low + np.average(boxes_high_aspect, axis=0) * high + np.average(
    #     boxes_normal_aspect, axis=0) * normal, np.array([b['boxes'] for b in filterd])
    if int(simu_points[i_gt, 2]) == 1:  # small object
        if len(filted_boxlist) >= Top_N_Boxes_small:
            return np.array([b['boxes'] for b in filted_boxlist[:Top_N_Boxes_small]]).reshape(
                (Top_N_Boxes_small, 4))
        else:
            return np.array([b['boxes'] for b in filted_boxlist[:len(filted_boxlist)]]).reshape(
                (len(filted_boxlist), 4))
    else:
        if len(filted_boxlist) >= Top_N_Boxes_big:
            return np.array([b['boxes'] for b in filted_boxlist[:Top_N_Boxes_big]]).reshape(
                (Top_N_Boxes_big, 4))
        else:
            return np.array([b['boxes'] for b in filted_boxlist[:len(filted_boxlist)]]).reshape(
                (len(filted_boxlist), 4))


def get_grid_bbox(im):
    grid_pixels = 10  # im.shape (rows,columns,channel) with respect to (height,width)
    grid_height_num = im.shape[0] / grid_pixels if im.shape[0] % grid_pixels == 0 else im.shape[0] / grid_pixels + 1
    grid_width_num = im.shape[1] / grid_pixels if im.shape[1] % grid_pixels == 0 else im.shape[1] / grid_pixels + 1
    img_grids = np.zeros((grid_height_num * grid_width_num, 4), dtype=np.uint16)
    index = 0
    for h in range(0, im.shape[0], grid_pixels):
        for w in range(0, im.shape[1], grid_pixels):
            img_grids[index] = [w, h,
                                w + grid_pixels if w <= im.shape[1] and im.shape[1] - w > grid_pixels else im.shape[
                                                                                                               1] - 1,
                                h + grid_pixels if h <= im.shape[0] and im.shape[0] - h > grid_pixels
                                else im.shape[0] - 1]
            index += 1
    return img_grids


def heat_map_bounding_boxes(d, roidb, boxlist):
    import matplotlib.pyplot as plt
    simu_gt_boxeslist = []
    view_result = False
    _t = {'total_time': Timer(), 'one_gt': Timer(),
          'filter_selective_boxlist': Timer(), 'get_grid_bbox': Timer(),
          'filter_grid': Timer(), 'bbox_overlaps_grid': Timer()}
    _t['total_time'].tic()
    for i_roi in xrange(len(roidb)):
        im = cv2.imread(d.image_path_at(i_roi))
        print 'image_index:', i_roi
        im = im[:, :, (2, 1, 0)]
        gt_boxes = roidb[i_roi]['boxes']
        gt_points = roidb[i_roi]['gt_points']
        simu_points = roidb[i_roi]['simu_points']
        simu_gt_boxes = np.zeros((len(gt_points), 4))
        _t['one_gt'].tic()
        for i_gt in xrange(len(gt_boxes)):
            gt_bbox = gt_boxes[i_gt, :4].astype(np.float)
            simu_point = simu_points[i_gt, :3]
            gt_point = gt_points[i_gt, :2]

            # show selective bounding box filtered
            # _t['filter_selective_boxlist'].tic()
            filted_boxlist = filter_selective_boxlist_heatmap(im, simu_points.astype(np.float),
                                                                                    i_gt, boxlist, i_roi)
            averge_boxes = np.average(filted_boxlist, axis=0)
            # averge_boxes, filted_boxlist = filter_selective_boxlist_heatmap(simu_points.astype(np.float), i_gt, boxlist,
            #                                                                 i_roi)
            simu_gt_boxes[i_gt, :] = [averge_boxes[0], averge_boxes[1], averge_boxes[2], averge_boxes[3]]

            # vis_results2(d, i_roi, gt_point, simu_point, gt_bbox, filted_boxlist, averge_boxes,
            #              view_result)
            '''
            _t['filter_selective_boxlist'].toc()
            # visualize iou heatmap
            _t['get_grid_bbox'].tic()
            im_grids = get_grid_bbox(im)
            filtered_box_l = np.array(filted_boxlist)
            _t['get_grid_bbox'].toc()
            _t['bbox_overlaps_grid'].tic()
            grid_overlaps = bbox_overlaps_grid(im_grids.astype(np.float), filtered_box_l.astype(np.float))
            _t['bbox_overlaps_grid'].toc()

            sum_grids = np.sum(grid_overlaps, axis=1)
            # print(im_grids.shape, grid_overlaps.shape, sum_grids.shape)
            _max = np.max(sum_grids)
            _min = np.min(sum_grids)
            # to do distance beteen grid and mid coords
            pred_flag_init = False
            if int(simu_point[2]) is 1:
                alpha_thresh = 0.7
                # plt.title('small_object')
            else:
                alpha_thresh = 0.3
                # plt.title('large_object')
            # print(im.shape,gt_bbox,1.0*(gt_bbox[2]-gt_bbox[0])*(gt_bbox[3]-gt_bbox[1])/(im.shape[0]*im.shape[1]))
            _t['filter_grid'].tic()
            for i_grid in range(len(im_grids)):
                grid = im_grids[i_grid]
                distance = get_distance(simu_point[0], simu_point[1], 0, 0, 0.5 *
                                        (grid[0] + grid[2]), 0.5 * (grid[1] + grid[3]))
                if int(simu_point[2]) is not 1:
                    alpha = (sum_grids[i_grid] - _min) / (_max - _min) * np.exp(-1.0 * distance / 800)
                else:
                    alpha = (sum_grids[i_grid] - _min) / (_max - _min) * np.cos(distance / 160)
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
            # ax.add_patch(
            #     plt.Rectangle((pred_xmin, pred_ymin),
            #                   pred_xmax - pred_xmin,
            #                   pred_ymax - pred_ymin, fill=False,
            #                   edgecolor='green', linewidth=3.5)
            # )
            if pred_xmax >= im.shape[1]:
                pred_xmax = pred_xmax - 1
            if pred_ymax >= im.shape[0]:
                pred_ymax = pred_ymax - 1
            _t['filter_grid'].toc()
            # if pred_xmin < 0 or pred_ymin < 0:
            #     print('pred_xmin:', pred_xmin, 'pred_ymin:', pred_ymin)
            simu_gt_boxes[i_gt, :4] = [pred_xmin, pred_ymin, pred_xmax, pred_ymax]  # gt_boxes[i,:]
        _t['one_gt'].toc()
        simu_gt_boxeslist.append(simu_gt_boxes)
        # print 'simu_gt_boxes',simu_gt_boxes
        # plt.axis('off')
        # plt.tight_layout()
        # plt.draw()
        # plt.show()
        # _t = {'total_time': Timer(), 'one_gt': Timer(),
        #       'filter_selective_boxlist': Timer(), 'get_grid_bbox': Timer(),
        #       'filter_grid': Timer}
        # print 'one_gt:  {:.3f}s filter_selective_boxlist : {:.3f}s get_grid_bbox : {:.3f}s ' \
        #       'bbox_overlaps_grid : {:.3f}s filter_grid :{:.3f}s' \
        #     .format(_t['one_gt'].average_time,_t['filter_selective_boxlist'].average_time,
        #             _t['get_grid_bbox'].average_time,
        #             _t['bbox_overlaps_grid'].average_time,_t['filter_grid'].average_time)
        '''
        simu_gt_boxeslist.append(simu_gt_boxes)
    cache_file = os.path.join(d.cache_path, d.name + '_gt_roidb_with_simu_bbox_heatmap_new.pkl')
    with open(cache_file, 'wb') as fid:
        cPickle.dump(simu_gt_boxeslist, fid, cPickle.HIGHEST_PROTOCOL)
    _t['total_time'].toc()
    print 'total_time:{:.3f}s'.format(_t['total_time'].average_time)


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


def calculate_corloc():
    cache_file_simu_points = os.path.join(_d.cache_path, 'voc_2007_trainval_gt_roidb_with_simu_points.pkl')
    if os.path.exists(cache_file_simu_points):
        with open(cache_file_simu_points, 'rb') as cache_file_simu_points_fid:
            roidb_simu_points = cPickle.load(cache_file_simu_points_fid)
            # print roidb_simu_points[0], len(roidb_simu_points[0])
            cache_file_simu_bbox = os.path.join(_d.cache_path,
                                                'voc_2007_trainval_gt_roidb_with_simu_bbox_heatmap.pkl')
            if os.path.exists(cache_file_simu_bbox):
                with open(cache_file_simu_bbox, 'rb') as cache_file_simu_bbox_fid:
                    roidb_simu_bbox = cPickle.load(cache_file_simu_bbox_fid)
                    # print roidb_simu_bbox[0], len(roidb_simu_bbox[0])
                    total_boxes = 0
                    boxes_over_thresh = 0
                    postive_picture = 0
                    objects = np.zeros((20, 2))
                    for index in xrange(len(roidb_simu_points)):
                        # print index
                        gt_bboxes = roidb_simu_points[index]['boxes']
                        simu_bboxes = roidb_simu_bbox[index]
                        assert np.shape(gt_bboxes) == np.shape(simu_bboxes)
                        overlaps = bbox_overlaps(gt_bboxes.astype(np.float), simu_bboxes.astype(np.float))

                        overlap_dia = overlaps.diagonal().copy()
                        gt_classes = roidb_simu_points[index]['gt_classes']
                        counted_class = []
                        for i_clss, gt_class_ in enumerate(gt_classes):
                            if int(gt_class_) not in counted_class:
                                if overlap_dia[i_clss] >= 0.5:
                                    objects[int(gt_class_) - 1, :] += 1
                                    counted_class.append(int(gt_class_))

                        for i_clss, gt_class_ in enumerate(gt_classes):
                            if int(gt_class_) not in counted_class:
                                objects[int(gt_class_) - 1, -1] += 1
                                counted_class.append(int(gt_class_))
                        # print overlaps,overlaps.diagonal(),len(overlaps.diagonal())
                        import warnings
                        warnings.filterwarnings('error')
                        try:
                            overlap_dia = overlaps.diagonal().copy()
                            # print overlap_dia
                            overlap_dia[overlap_dia >= 0.5] = 1
                            overlap_dia[overlap_dia < 0.5] = 0
                        except RuntimeWarning as rw:
                            print overlap_dia
                            print gt_bboxes
                            print simu_bboxes
                            print index
                            print rw
                        total_boxes += len(gt_bboxes)
                        boxes_over_thresh += np.size(np.nonzero(overlap_dia))
                        postive_picture += 1 if np.size(np.nonzero(overlap_dia)) is not 0 else 0
                        assert len(gt_bboxes) == len(simu_bboxes)
                    print 'LocCor=', boxes_over_thresh, total_boxes, 1.0 * boxes_over_thresh / total_boxes
                    print 'LocCor2=', postive_picture, len(roidb_simu_points), 1.0 * postive_picture / len(
                        roidb_simu_points)
                    print objects
                    print np.sum(objects,axis=0)
                    print 'true LocCor:', objects[:, 0] / objects[:, 1], np.mean(objects[:, 0] / objects[:, 1])


def calculate_positive_boxes(boxes, rpn_bboxes):
    assert np.shape(boxes) == np.shape(rpn_bboxes['rpn_simu_gt_boxes'][:, 0:4])
    overlaps = bbox_overlaps(boxes.astype(np.float),
                             rpn_bboxes['rpn_simu_gt_boxes'][:, 0:4].astype(np.float))
    overlap_dia = overlaps.diagonal().copy()
    # overlap_dia[overlap_dia >= 0.95] = 1
    overlap_dia[overlap_dia < 0.5] = 0
    return np.size(np.nonzero(overlap_dia))


def calculate_corloc_rpn(d):
    cache_file_simu_points = os.path.join(_d.cache_path, 'voc_2007_trainval_gt_roidb_with_simu_points.pkl')
    cache_file_simu_bbox = os.path.join(_d.cache_path, 'voc_2007_trainval_gt_roidb_with_simu_bbox_heatmap.pkl')
    rpn_file_list = glob.glob(_d.cache_path + '/voc_2007_trainval_rpn_pse_gt_boxes_classifi_bg_valid*.pkl')
    view_result = False
    if os.path.exists(cache_file_simu_points):
        with open(cache_file_simu_points, 'rb') as cache_file_simu_points_fid, open(cache_file_simu_bbox,
                                                                                    'rb') as cache_file_simu_bbox_fid:
            roidb_simu_points = cPickle.load(cache_file_simu_points_fid)
            roidb_simu_boxes = cPickle.load(cache_file_simu_bbox_fid)
            for rpn_file in sorted(rpn_file_list, key=os.path.getmtime):
                rpn_file_bbox = os.path.join(rpn_file)
                if os.path.exists(rpn_file_bbox):
                    with open(rpn_file_bbox, 'rb') as rpn_bbox_fid:
                        roidb_rpn_boxes = cPickle.load(rpn_bbox_fid)
                        total_boxes = 0
                        boxes_over_thresh = 0
                        for index in xrange(len(roidb_simu_points)):
                            rpn_bboxes1 = roidb_rpn_boxes[2 * index]
                            rpn_bboxes2 = roidb_rpn_boxes[2 * index + 1]

                            assert rpn_bboxes1['img_index'] == rpn_bboxes2['img_index']
                            img_path = d.image_path_at(index)
                            gt_bboxes = roidb_simu_points[index]['boxes']
                            simu_bboxes = roidb_simu_boxes[index]
                            assert len(gt_bboxes) == len(rpn_bboxes1['rpn_simu_gt_boxes'])
                            assert len(gt_bboxes) == len(rpn_bboxes2['rpn_simu_gt_boxes'])
                            # width = PIL.Image.open(img_path).size[0]
                            im = cv2.imread(img_path)
                            width = im.shape[1]
                            im = im[:, :, (2, 1, 0)]
                            total_boxes += len(gt_bboxes) * 2
                            try:
                                # if (rpn_bboxes1['rpn_simu_gt_boxes'][:, 4].astype(np.int) == 1).all():
                                if rpn_bboxes1['flipped'] == 1:
                                    # flip boxes
                                    gt_boxes_ = gt_bboxes.copy()
                                    # print boxes
                                    oldx1 = gt_boxes_[:, 0].copy()
                                    oldx2 = gt_boxes_[:, 2].copy()
                                    gt_boxes_[:, 0] = width - oldx2 - 1
                                    gt_boxes_[:, 2] = width - oldx1 - 1

                                    simu_bboxes_ = simu_bboxes.copy()
                                    oldx1 = simu_bboxes_[:, 0].copy()
                                    oldx2 = simu_bboxes_[:, 2].copy()
                                    simu_bboxes_[:, 0] = width - oldx2 - 1
                                    simu_bboxes_[:, 2] = width - oldx1 - 1
                                    # print boxes
                                    boxes_over_thresh += calculate_positive_boxes(gt_boxes_, rpn_bboxes1)
                                    vis_results(im[:, ::-1, :], gt_boxes_, rpn_bboxes1, simu_bboxes_, view_result)
                                else:
                                    boxes_over_thresh += calculate_positive_boxes(gt_bboxes, rpn_bboxes1)
                                    vis_results(im, gt_bboxes, rpn_bboxes1, simu_bboxes, view_result)
                                # if (rpn_bboxes2['rpn_simu_gt_boxes'][:, 4].astype(np.int) == 1).all():
                                if rpn_bboxes2['flipped'] == 1:
                                    # flip boxes
                                    gt_boxes_ = gt_bboxes.copy()
                                    oldx1 = gt_boxes_[:, 0].copy()
                                    oldx2 = gt_boxes_[:, 2].copy()
                                    gt_boxes_[:, 0] = width - oldx2 - 1
                                    gt_boxes_[:, 2] = width - oldx1 - 1

                                    simu_bboxes_ = simu_bboxes.copy()
                                    oldx1 = simu_bboxes_[:, 0].copy()
                                    oldx2 = simu_bboxes_[:, 2].copy()
                                    simu_bboxes_[:, 0] = width - oldx2 - 1
                                    simu_bboxes_[:, 2] = width - oldx1 - 1
                                    boxes_over_thresh += calculate_positive_boxes(gt_boxes_, rpn_bboxes2)
                                    vis_results(im[:, ::-1, :], gt_boxes_, rpn_bboxes2, simu_bboxes_, view_result)
                                else:
                                    boxes_over_thresh += calculate_positive_boxes(gt_bboxes, rpn_bboxes2)
                                    vis_results(im, gt_bboxes, rpn_bboxes2, simu_bboxes, view_result)
                            except RuntimeWarning as rw:
                                print gt_bboxes
                                print rpn_bboxes1
                                print index
                                print rw
                        print rpn_file_bbox[-10:], ',LocCor=', boxes_over_thresh, ',', total_boxes, ',', \
                            1.0 * boxes_over_thresh / total_boxes


def vis_results2(d, i_roi, gt_point, simu_point, gt_bbox, filted_boxlist, full_filterd_boxlist, averge_boxes, show):
    import matplotlib.pyplot as plt
    if show:
        im = cv2.imread(d.image_path_at(i_roi))
        print 'image_index:', i_roi
        im = im[:, :, (2, 1, 0)]
        fig, ax = plt.subplots(figsize=(12, 12))
        ax.imshow(im, aspect='equal')

        ax.add_patch(
            plt.Rectangle((gt_bbox[0], gt_bbox[1]),
                          gt_bbox[2] - gt_bbox[0],
                          gt_bbox[3] - gt_bbox[1], fill=False,
                          edgecolor='red', linewidth=3.5)
        )
        plt.plot(simu_point[0], simu_point[1], marker='*', markersize=8)
        plt.plot(gt_point[0], gt_point[1], marker='o', markersize=8)
        for filter_box in filted_boxlist:
            ax.add_patch(
                plt.Rectangle((filter_box[0], filter_box[1]),
                              filter_box[2] - filter_box[0],
                              filter_box[3] - filter_box[1], fill=False,
                              edgecolor='green', linewidth=1.5)
            )
        ax.add_patch(
            plt.Rectangle((averge_boxes[0], averge_boxes[1]),
                          averge_boxes[2] - averge_boxes[0],
                          averge_boxes[3] - averge_boxes[1], fill=False,
                          edgecolor='yellow', linewidth=1.5)
        )

        overlap_bbox_each = []
        for index_i, temp_i in enumerate(full_filterd_boxlist):
            print 'first:', index_i, temp_i
            for _, temp_j in enumerate(full_filterd_boxlist[index_i + 1:]):
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
        assert len(overlap_bbox_each) == len(full_filterd_boxlist) * (len(full_filterd_boxlist) - 1) / 2
        overlap_average = np.average(np.array(overlap_bbox_each), axis=0)
        # print overlap_average
        ax.add_patch(
            plt.Rectangle((overlap_average[0], overlap_average[1]),
                          overlap_average[2] - overlap_average[0],
                          overlap_average[3] - overlap_average[1], fill=False,
                          edgecolor='blue', linewidth=1)
        )
        plt.axis('off')
        plt.tight_layout()
        plt.title('small_obj:' + str(simu_point[2]) + ',boxes:' + str(len(filted_boxlist)))
        plt.draw()
        plt.show()


def vis_results(im, boxes, rpn_bboxes, simu_boxes, isshow):
    import matplotlib.pyplot as plt
    if isshow:
        fig, ax = plt.subplots(figsize=(12, 12))
        ax.imshow(im, aspect='equal')
        for box in boxes:
            ax.add_patch(
                plt.Rectangle((box[0], box[1]),
                              box[2] - box[0],
                              box[3] - box[1], fill=False,
                              edgecolor='green', linewidth=1.5)
            )
        for rpn_box in rpn_bboxes['rpn_simu_gt_boxes']:
            ax.add_patch(
                plt.Rectangle((rpn_box[0], rpn_box[1]),
                              rpn_box[2] - rpn_box[0],
                              rpn_box[3] - rpn_box[1], fill=False,
                              edgecolor='red', linewidth=1.5)
            )
        for simu_box in simu_boxes:
            ax.add_patch(
                plt.Rectangle((simu_box[0], simu_box[1]),
                              simu_box[2] - simu_box[0],
                              simu_box[3] - simu_box[1], fill=False,
                              edgecolor='yellow', linewidth=1.5)
            )
        plt.axis('off')
        plt.tight_layout()
        plt.title('green:GT, red: rpn_simu, yellow: SS')
        plt.draw()
        plt.show()


if __name__ == '__main__':
    # from datasets.pascal_voc import pascal_voc
    _d = pascal_voc('trainval', '2007')

    # res = _d.roidb
    # from IPython import embed; embed()
    dataset_name = 'voc_2007_trainval'
    _roidb = gt_roidb_with_simu_points('/home/leochang/Downloads/PycharmProjects/py-faster-rcnn/data/cache',
                                       dataset_name)
    _boxlist = load_selective_search_roidb(dataset_name)
    # average_bounding_boxes(_d, _roidb, _boxlist)
    # heat_map_bounding_boxes(_d, _roidb, _boxlist)
    # calculate_corloc()
    calculate_corloc_rpn(_d)
