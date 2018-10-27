# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick,modified by leo chang
# this file can be used to get middle coordinates of each ground truth bounding box
# --------------------------------------------------------

import datasets
import datasets.pascal_voc
import os
import datasets.imdb
import xml.dom.minidom as minidom
import numpy as np
import scipy.sparse
import scipy.io as sio
import utils.cython_bbox
import cPickle
import subprocess
import matplotlib
import matplotlib.pyplot as plt
import cv2
from utils.cython_bbox_grid import bbox_overlaps_grid
from utils.cython_bbox import bbox_overlaps
small_object = 1
median_object = 2
large_object = 3
big_aspect_object = 4

SMALL_OBJECT_TO_BBX_CENETER = 9
BIG_OBJECT_TO_BBX_CENTER = 20

class pascal_voc_gt_mid(datasets.imdb):
    def __init__(self, image_set, year, devkit_path=None):
        datasets.imdb.__init__(self, 'voc_' + year + '_' + image_set)
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
        self._roidb_handler = self.selective_search_roidb

        # PASCAL specific config options
        self.config = {'cleanup': True,
                       'use_salt': True,
                       'top_k': 2000}

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
        return os.path.join(datasets.ROOT_DIR, 'data', 'VOCdevkit' + self._year)

    def gt_mid_coords_roidb(self, name):
        """
        Return the mid coords database of ground-truth regions of interest.

        This function loads/saves from/to a cache file to speed up future calls.
        """
        if name is 'random_all_area_mid_coords_roidb':
            cache_file = os.path.join(self.cache_path, self.name + '_gt_random_all_area_mid_coords_roidb.pkl')
        elif name is 'mid_coords_roidb':
            cache_file = os.path.join(self.cache_path, self.name + '_gt_mid_coords_roidb.pkl')
        elif name is 'random_mid_2_coords_roidb':
            cache_file = os.path.join(self.cache_path, self.name + '_gt_random_mid_2_coords_roidb.pkl')
        elif name is 'simulated_center':
            cache_file = os.path.join(self.cache_path,self.name+'_simulated_center_with_class.pkl')
        else:
            cache_file = os.path.join(self.cache_path, self.name + '_gt_random_mid_coords_roidb.pkl')
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                roidb = cPickle.load(fid)
            print '{} gt roidb loaded from {}'.format(self.name, cache_file)
            return roidb

        gt_roidb = [self._load_pascal_annotation_mid_coordinates(index)
                    for index in self.image_index]
        with open(cache_file, 'wb') as fid:
            cPickle.dump(gt_roidb, fid, cPickle.HIGHEST_PROTOCOL)
        print 'wrote gt roidb to {}'.format(cache_file)

        return gt_roidb

    def gt_roidb(self):
        """
        Return the database of ground-truth regions of interest.

        This function loads/saves from/to a cache file to speed up future calls.
        """
        cache_file = os.path.join(self.cache_path, self.name + '_gt_roidb.pkl')
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
            roidb = datasets.imdb.merge_roidbs(gt_roidb, ss_roidb)
        else:
            roidb = self._load_selective_search_roidb(None)
        with open(cache_file, 'wb') as fid:
            cPickle.dump(roidb, fid, cPickle.HIGHEST_PROTOCOL)
        print 'wrote ss roidb to {}'.format(cache_file)

        return roidb

    def load_ss_roidb(self):
        '''
        load selective search roidb from file
        :return:
        '''
        filename = os.path.abspath(os.path.join(self.cache_path, '..',
                                                'selective_search_data',
                                                self.name + '.mat'))
        assert os.path.exists(filename), \
            'Selective search data not found at: {}'.format(filename)
        raw_data = sio.loadmat(filename)['boxes'].ravel()

        box_list = []
        for i in xrange(raw_data.shape[0]):
            # exchange xy coordinates
            box_list.append(raw_data[i][:, (1, 0, 3, 2)] - 1)

        # print len(box_list), len(box_list[1]), box_list[1][2]
        return box_list

    def _load_selective_search_roidb(self, gt_roidb):
        filename = os.path.abspath(os.path.join(self.cache_path, '..',
                                                'selective_search_data',
                                                self.name + '.mat'))
        assert os.path.exists(filename), \
            'Selective search data not found at: {}'.format(filename)
        raw_data = sio.loadmat(filename)['boxes'].ravel()

        box_list = []
        for i in xrange(raw_data.shape[0]):
            box_list.append(raw_data[i][:, (1, 0, 3, 2)] - 1)

        return self.create_roidb_from_box_list(box_list, gt_roidb)

    def selective_search_IJCV_roidb(self):
        """
        Return the database of selective search regions of interest.
        Ground-truth ROIs are also included.

        This function loads/saves from/to a cache file to speed up future calls.
        """
        cache_file = os.path.join(self.cache_path,
                                  '{:s}_selective_search_IJCV_top_{:d}_roidb.pkl'.
                                  format(self.name, self.config['top_k']))

        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                roidb = cPickle.load(fid)
            print '{} ss roidb loaded from {}'.format(self.name, cache_file)
            return roidb

        gt_roidb = self.gt_roidb()
        ss_roidb = self._load_selective_search_IJCV_roidb(gt_roidb)
        roidb = datasets.imdb.merge_roidbs(gt_roidb, ss_roidb)
        with open(cache_file, 'wb') as fid:
            cPickle.dump(roidb, fid, cPickle.HIGHEST_PROTOCOL)
        print 'wrote ss roidb to {}'.format(cache_file)

        return roidb

    def _load_selective_search_IJCV_roidb(self, gt_roidb):
        IJCV_path = os.path.abspath(os.path.join(self.cache_path, '..',
                                                 'selective_search_IJCV_data',
                                                 'voc_' + self._year))
        assert os.path.exists(IJCV_path), \
            'Selective search IJCV data not found at: {}'.format(IJCV_path)

        top_k = self.config['top_k']
        box_list = []
        for i in xrange(self.num_images):
            filename = os.path.join(IJCV_path, self.image_index[i] + '.mat')
            raw_data = sio.loadmat(filename)
            box_list.append((raw_data['boxes'][:top_k, :] - 1).astype(np.uint16))

        return self.create_roidb_from_box_list(box_list, gt_roidb)

    def _load_pascal_annotation_mid_coordinates(self, index):
        '''
        load bouding boxes and calculate each mid coordinates of groud truth
        info from XML file in the PASCAL VOC format
        :param index:
        :return:
        '''
        filename = os.path.join(self._data_path, 'Annotations', index + '.xml')
        use_diff = True

        # print 'Loading: {}'.format(filename)
        def get_data_from_tag(node, tag):
            return node.getElementsByTagName(tag)[0].childNodes[0].data

        with open(filename) as f:
            data = minidom.parseString(f.read())

        objs = data.getElementsByTagName('object')
        print('______len(objs)_______', len(objs))
        if not use_diff:
            non_diff_objs = [
                obj for obj in objs if int(get_data_from_tag(obj, 'difficult')) == 0]
        # objs = non_diff_objs
        print('______len(objs)_______', len(objs))
        size = data.getElementsByTagName('size')
        img_width = float(get_data_from_tag(size[0], 'width'))
        img_height = float(get_data_from_tag(size[0], 'height'))
        num_objs = len(objs)
        boxes = np.zeros((num_objs, 4), dtype=np.uint16)
        mid_coors = np.zeros((num_objs, 4), dtype=np.uint16)
        gt_classes = np.zeros((num_objs), dtype=np.int32)
        np.random.seed(2)
        # Load object bounding boxes into a data frame.
        for ix, obj in enumerate(objs):
            # Make pixel indexes 0-based
            x1 = float(get_data_from_tag(obj, 'xmin')) - 1
            y1 = float(get_data_from_tag(obj, 'ymin')) - 1
            x2 = float(get_data_from_tag(obj, 'xmax')) - 1
            y2 = float(get_data_from_tag(obj, 'ymax')) - 1
            mid_x = (x1 + x2) / 2 #get_random_mid_coords(x1, x2)
            mid_y = (y1 + y2) / 2 #get_random_mid_coords(y1, y2)  #
            # mid_x_left, mid_y_up, mid_x_right, mid_y_down, object_size = get_random_mid_coords(x1, x2, y1, y2,
            #                                                                                    img_width, img_height)
            cls = self._class_to_ind[
                str(get_data_from_tag(obj, "name")).lower().strip()]
            boxes[ix, :] = [x1, y1, x2, y2]
            mid_coors[ix, :] = [mid_x,mid_y,img_width,img_height]#[mid_x_left, mid_y_up, mid_x_right, mid_y_down, object_size]
            gt_classes[ix] = cls
        return {'boxes': boxes,
                'mid_coors': mid_coors,
                'gt_classes': gt_classes,
                'index': index, }

    def _load_pascal_annotation(self, index):
        """
        Load image and bounding boxes info from XML file in the PASCAL VOC
        format.
        """
        filename = os.path.join(self._data_path, 'Annotations', index + '.xml')

        # print 'Loading: {}'.format(filename)
        def get_data_from_tag(node, tag):
            return node.getElementsByTagName(tag)[0].childNodes[0].data

        with open(filename) as f:
            data = minidom.parseString(f.read())

        objs = data.getElementsByTagName('object')
        num_objs = len(objs)

        boxes = np.zeros((num_objs, 4), dtype=np.uint16)
        gt_classes = np.zeros((num_objs), dtype=np.int32)
        overlaps = np.zeros((num_objs, self.num_classes), dtype=np.float32)

        # Load object bounding boxes into a data frame.
        for ix, obj in enumerate(objs):
            # Make pixel indexes 0-based
            x1 = float(get_data_from_tag(obj, 'xmin')) - 1
            y1 = float(get_data_from_tag(obj, 'ymin')) - 1
            x2 = float(get_data_from_tag(obj, 'xmax')) - 1
            y2 = float(get_data_from_tag(obj, 'ymax')) - 1
            cls = self._class_to_ind[
                str(get_data_from_tag(obj, "name")).lower().strip()]
            boxes[ix, :] = [x1, y1, x2, y2]
            gt_classes[ix] = cls
            overlaps[ix, cls] = 1.0

        overlaps = scipy.sparse.csr_matrix(overlaps)

        return {'boxes': boxes,
                'gt_classes': gt_classes,
                'gt_overlaps': overlaps,
                'flipped': False}

    def _write_voc_results_file(self, all_boxes):
        use_salt = self.config['use_salt']
        comp_id = 'comp4'
        if use_salt:
            comp_id += '-{}'.format(os.getpid())

        # VOCdevkit/results/VOC2007/Main/comp4-44503_det_test_aeroplane.txt
        path = os.path.join(self._devkit_path, 'results', 'VOC' + self._year,
                            'Main', comp_id + '_')
        for cls_ind, cls in enumerate(self.classes):
            if cls == '__background__':
                continue
            print 'Writing {} VOC results file'.format(cls)
            filename = path + 'det_' + self._image_set + '_' + cls + '.txt'
            with open(filename, 'wt') as f:
                for im_ind, index in enumerate(self.image_index):
                    dets = all_boxes[cls_ind][im_ind]
                    if dets == []:
                        continue
                    # the VOCdevkit expects 1-based indices
                    for k in xrange(dets.shape[0]):
                        f.write('{:s} {:.3f} {:.1f} {:.1f} {:.1f} {:.1f}\n'.
                                format(index, dets[k, -1],
                                       dets[k, 0] + 1, dets[k, 1] + 1,
                                       dets[k, 2] + 1, dets[k, 3] + 1))
        return comp_id

    def evaluate_detections(self, all_boxes, output_dir):
        comp_id = self._write_voc_results_file(all_boxes)
        self._do_matlab_eval(comp_id, output_dir)

    def competition_mode(self, on):
        if on:
            self.config['use_salt'] = False
            self.config['cleanup'] = False
        else:
            self.config['use_salt'] = True
            self.config['cleanup'] = True

    def vis_detections(self, im, object_simulated_center, object_mid_coords, box_list,index):
        """Draw detected bounding boxes."""

        im = im[:, :, (2, 1, 0)]
        gt_boxes = object_mid_coords[index]['boxes']
        simu_centers = object_simulated_center[index]['simlu_centers']
        coords_mid = object_mid_coords[index]['mid_coors']
        gt_classes = object_mid_coords[index]['gt_classes']
        # ax.imshow(im, aspect='equal')
        # show ground truth bounding box and mid coords
        pred_boxes = np.zeros((len(gt_boxes), 4), dtype=np.uint16)
        pred_gt_classes = np.zeros(len(gt_boxes), dtype=np.int32)
        pred_seg_areas = np.zeros((len(gt_boxes)), dtype=np.float32)
        corLoc = np.zeros((len(gt_boxes),4),dtype=np.float32)
        # simu_centers = np.zeros((len(gt_boxes),4),dtype=np.uint32)
        # fig, ax = plt.subplots(figsize=(12, 12))
        # ax.imshow(im, aspect='equal')
        pred_overlaps = np.zeros((len(gt_boxes), self.num_classes), dtype=np.float32)

        for i in range(len(gt_boxes)):
            filtered_box_list, counter = [], 0
            cls = gt_classes[i]
            # fig, ax = plt.subplots(figsize=(12, 12))
            # ax.imshow(im, aspect='equal')
            gt_bbox = gt_boxes[i, :4].astype(np.float)
            simu_cds = simu_centers[i, :4]
            mid_cds2 = coords_mid[i, :2]

            # generate simulated center points
            is_small_object = False
            if (gt_bbox[3]-gt_bbox[1])*(gt_bbox[2]-gt_bbox[0])/(im.shape[0]*im.shape[1]) <= 0.1:
                is_small_object = True
            print((gt_bbox[3]-gt_bbox[1])*(gt_bbox[2]-gt_bbox[0])/(im.shape[0]*im.shape[1]),is_small_object)
            simu_x,simu_y = generate_simulated_center(gt_bbox[0],gt_bbox[1],is_small_object)
            simu_centers[i,:]=[simu_x,simu_y,cls,int(is_small_object)]
            '''
            ax.add_patch(
                plt.Rectangle((gt_bbox[0], gt_bbox[1]),
                              gt_bbox[2] - gt_bbox[0],
                              gt_bbox[3] - gt_bbox[1],  fill=False,
                              edgecolor='red', linewidth=3.5)
            )
            plt.plot(simu_cds[0], simu_cds[1], marker='*', markersize=12)
            # plt.plot(cds[2], cds[3], marker='*', markersize=12)
            plt.plot(mid_cds2[0], mid_cds2[1], marker='o', markersize=12)
            # plt.plot(simu_x,simu_y, marker='o', markersize=12)
            print(cls,self._classes[cls])
            # show selective bounding box filtered

            for box_index in range(0, len(box_list), 1):
                bbox = box_list[box_index, :4]
                # for cds in coords:
                # cds = coords[0]
                # filter out some boxes that don't contain any simulated middle coordinates

                pixels_thresh = 20  # for robustness to add some pixels
                if bbox[0] - pixels_thresh < simu_cds[0] and bbox[1] - pixels_thresh < simu_cds[1] \
                        and bbox[2] + pixels_thresh > simu_cds[0] and bbox[3] + pixels_thresh > simu_cds[1]:

                    # for two simulated points
                    # if int(simu_cds[2]) is not 0 and int(simu_cds[3]) is not 0:
                    #     if bbox[0] - pixels_thresh < simu_cds[2] and bbox[1] - pixels_thresh < simu_cds[3] \
                    #             and bbox[2] + pixels_thresh > simu_cds[2] and bbox[3] + pixels_thresh > simu_cds[3]:
                    #
                    #         if bbox not in filtered_box_list:
                    #             filtered_box_list.append(bbox)
                    #         else:
                    #             print('bbox in filtered_box_list')
                    # else:
                    # filtered_box_list.index(bbox)
                    # print([(box - bbox).all() for box in filtered_box_list])
                    # if True not in [(box - bbox).all() for box in filtered_box_list]:
                    filtered_box_list.append(bbox)
                    # else:
                    #     print('bbox in filtered_box_list')
                    # ax.add_patch(
                    #     plt.Rectangle((bbox[0], bbox[1]),
                    #                   bbox[2] - bbox[0],
                    #                   bbox[3] - bbox[1],
                    #                   fill=False,
                    #                   edgecolor='blue', linewidth=1.5)
                    # )
                    counter += 1
                    # break

            print(len(box_list),len(filtered_box_list),counter)
            # plt.axis('off')
            # plt.tight_layout()
            # plt.draw()
            # plt.show()
            # visualize iou heatmap
            im_grids = get_grid_bbox(im)
            filtered_box_l = np.array(filtered_box_list)
            grid_overlaps = bbox_overlaps_grid(im_grids.astype(np.float), filtered_box_l.astype(np.float))
            sum_grids = np.sum(grid_overlaps, axis=1)
            # print(im_grids.shape, grid_overlaps.shape, sum_grids.shape)
            _max = np.max(sum_grids)
            _min = np.min(sum_grids)

            # to do distance beteen grid and mid coords

            pred_xmin, pred_xmax, pred_ymin, pred_ymax = 0, 0, 0, 0
            pred_flag_init = False
            if int(simu_cds[3]) is small_object:
                alpha_thresh = 0.35
                plt.title('small_object')
            # elif int(cds[4]) is median_object:
            #     alpha_thresh = 0.4
            #     # plt.title('median_object')
            # elif int(cds[4]) is large_object or int(cds[4]) is big_aspect_object:
            else:
                alpha_thresh = 0.3
                plt.title('large_object')
            print(simu_cds,simu_cds[3])
            print(im.shape,gt_bbox,1.0*(gt_bbox[2]-gt_bbox[0])*(gt_bbox[3]-gt_bbox[1])/(im.shape[0]*im.shape[1]))
            for id in range(len(im_grids)):
                grid = im_grids[id]
                # distance = np.linalg.norm()

                if int(simu_cds[3]) is not small_object:
                    distance = get_distance(simu_cds[0], simu_cds[1], 0, 0, 0.5 *
                                            (grid[0] + grid[2]), 0.5 * (grid[1] + grid[3]))
                    alpha = (sum_grids[id] - _min) / (_max - _min) * np.exp(-1.0 * distance / 800)
                # elif int(cds[4]) is median_object:
                #     distance = get_distance(cds[0], cds[1], cds[2], cds[3], 0.5 *
                #                             (grid[0] + grid[2]), 0.5 * (grid[1] + grid[3]))
                #     alpha = (sum_grids[id] - _min) / (_max - _min) * np.exp(-1.0 * distance / 800)
                else:
                    alpha = (sum_grids[id] - _min) / (_max - _min)

                if alpha >= alpha_thresh:
                    ax.add_patch(
                        plt.Rectangle((grid[0], grid[1]),
                                      grid[2] - grid[0],
                                      grid[3] - grid[1], fill=True,
                                      color='yellow', alpha=alpha, edgecolor=None)
                    )
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
            ax.add_patch(
                plt.Rectangle((pred_xmin, pred_ymin),
                              pred_xmax - pred_xmin,
                              pred_ymax - pred_ymin, fill=False,
                              edgecolor='green', linewidth=3.5)
            )
            if pred_xmax >= im.shape[0]:
                pred_xmax = pred_xmax - 1
            if pred_ymax >= im.shape[1]:
                pred_ymax = pred_ymax - 1

            if pred_xmin < 0 or pred_ymin < 0:
                print('pred_xmin:', pred_xmin, 'pred_ymin:', pred_ymin)
            pred_boxes[i, :] = [pred_xmin, pred_ymin, pred_xmax, pred_ymax]  # gt_boxes[i,:]
            pred_gt_classes[i] = cls
            pred_overlaps[i, cls] = 1.0
            pred_seg_areas[i] = (pred_xmax - pred_xmin + 1) * (pred_ymax - pred_ymin + 1)
            corLoc[i] = calculate_CorLoc(pred_boxes[i,:], gt_boxes[i, :4])

            plt.axis('off')
            plt.tight_layout()
            plt.draw()
            plt.show()
        pred_overlaps = scipy.sparse.csr_matrix(pred_overlaps)


        '''
        '''    
        return {'boxes': pred_boxes,
                'gt_classes': pred_gt_classes,
                'gt_overlaps': pred_overlaps,
                'flipped': False,
                'seg_areas': pred_seg_areas,
                'corLoc':corLoc}
        # return {'boxes':gt_boxes,
        #         'gt_classes':gt_classes,
        #         'simlu_centers':simu_centers,
        #         'filter_length':len(filtered_box_list)}
        '''
        return {'simlu_centers':simu_centers}

        # calculate the distance of between each mid coordinates and four coordinates
        #  of candidate boxes,and select top K min distance, K = 5 as a temporal choice
        # K = 50
        # selected_box_list = [[] for _ in range(len(coords[0]))]
        # for box in filtered_box_list:
        #     box = box * 1.0
        #     for index, cds in enumerate(coords[0]):
        #         cds = cds * 1.0
        #         distance = np.linalg.norm(np.array([box[0], box[1]]) - cds) + \
        #                    np.linalg.norm(np.array([box[0], box[1] + box[3]]) - cds) + \
        #                    np.linalg.norm(np.array([box[0] + box[2], box[1]]) - cds) + \
        #                    np.linalg.norm(np.array([box[0] + box[2], box[1] + box[3]]) - cds)
        #         if len(selected_box_list[index]) < K:
        #             selected_box_list[index].append([box, distance])
        #         else:
        #             dists = [dis[1] for dis in selected_box_list[index]]
        #             if distance < np.max(dists):
        #                 a = np.argmax(dists)
        #                 selected_box_list[index].pop(np.argmax(dists))
        #                 selected_box_list[index].append([box, distance])
        #
        # for ssboxes in selected_box_list:
        #     for bboxes in ssboxes:
        #         bbox = bboxes[0]
        #         ax.add_patch(
        #             plt.Rectangle((bbox[0], bbox[1]),
        #                           bbox[2],
        #                           bbox[3], fill=False,
        #                           edgecolor='green', linewidth=1.5)
        #         )
        #     ax.add_patch(
        #         plt.Rectangle((bbox[0], bbox[1]),
        #                       bbox[2],
        #                       bbox[3], fill=False,
        #                       edgecolor='green', linewidth=1.5)
        #     )


def calculate_CorLoc(pred_boxes,gt_boxes):
    box_area = (
            (gt_boxes[2] - gt_boxes[0] + 1) *
            (gt_boxes[3] - gt_boxes[1] + 1)
    )
    iw = (
            min(pred_boxes[2], gt_boxes[2]) -
            max(pred_boxes[0], gt_boxes[0]) + 1
    )
    if iw > 0:
        ih = (
                min(pred_boxes[3], gt_boxes[3]) -
                max(pred_boxes[1], gt_boxes[1]) + 1
        )
        if ih > 0:
            ua = float((pred_boxes[2] - pred_boxes[0] + 1) *
                           (pred_boxes[3] - pred_boxes[1] + 1) +
                        box_area - iw * ih)
            overlap = iw * ih / ua
            # print('overlap:',overlap)
            if overlap >= 0.5:
                return 1
            else:
                return 0
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


def get_grid_bbox(im):
    grid_pixels = 5
    img_grid_width = im.shape[0] / grid_pixels if im.shape[0] % grid_pixels == 0 else im.shape[0] / grid_pixels + 1
    img_grid_height = im.shape[1] / grid_pixels if im.shape[1] % grid_pixels == 0 else im.shape[1] / grid_pixels + 1
    img_grids = np.zeros((img_grid_width * img_grid_height, 4), dtype=np.uint16)
    index = 0
    for i in range(0, im.shape[0], grid_pixels):
        for j in range(0, im.shape[1], grid_pixels):
            img_grids[index] = [j, i,
                                j + grid_pixels if j <= im.shape[1] and im.shape[1] - j > grid_pixels else im.shape[
                                                                                                               1] - 1,
                                i + grid_pixels if i <= im.shape[0] and im.shape[0] - i > grid_pixels
                                else im.shape[0] - 1]
            index += 1
    return img_grids


def get_random_mid_coords(x1, x2, y1, y2, img_w, img_h):
    # np.random.seed(2)
    small_object_thresh = 0.1
    median_object_thresh = 0.4
    aspect_ratio_thresh = 0.55
    area_ratio = (x2 - x1) * (y2 - y1) / (img_w * img_h)

    # if object is  small, sample one point is enough,alpha threshold should be big
    if area_ratio < small_object_thresh:
        # if bounding box has big aspect ratio
        if (x2 - x1) / (y2 - y1) < aspect_ratio_thresh or (y2 - y1) / (x2 - x1) < aspect_ratio_thresh:
            if x2 - x1 > y2 - y1:
                avg_x = (x1 + x2) / 2
                # mid left point
                avg_x_left = (x1 + avg_x) / 2
                m = avg_x_left - x1
                t = x1 / m
                mid_x_left = m * (np.random.rand() + t)

                # mid right point
                avg_x_right = (x2 + avg_x) / 2
                m = x2 - avg_x_right
                t = avg_x_right / m
                mid_x_right = m * (np.random.rand() + t)

                mid_y = (y1 + y2) / 2
                return mid_x_left, mid_y, mid_x_right, mid_y, big_aspect_object
            else:
                avg_y = (y1 + y2) / 2
                # mid up point
                avg_y_up = (y1 + avg_y) / 2
                m = avg_y_up - y1
                t = y1 / m
                mid_y_up = m * (np.random.rand() + t)
                # mid down point
                avg_y_down = (y2 + avg_y) / 2
                m = y2 - avg_y_down
                t = avg_y_down / m
                mid_y_down = m * (np.random.rand() + t)

                mid_x = (x1 + x2) / 2
                return mid_x, mid_y_up, mid_x, mid_y_down, big_aspect_object
        else:
            avg_x = (x1 + x2) / 2
            x1 = (x1 + avg_x) / 2
            x2 = (x2 + avg_x) / 2
            m = x2 - x1
            t = x1 / m
            mid_x = m * (np.random.rand() + t)

            avg_y = (y1 + y2) / 2
            y1 = (y1 + avg_y) / 2
            y2 = (y2 + avg_y) / 2
            m = y2 - y1
            t = y1 / m
            mid_y = m * (np.random.rand() + t)

            # if mid_x > (x1 + x2) / 2:
            #     print(x1, x2, mid_x)
            return mid_x, mid_y, 0, 0, small_object
    else:
        # if bounding box has big aspect ratio
        if (x2 - x1) / (y2 - y1) < aspect_ratio_thresh or (y2 - y1) / (x2 - x1) < aspect_ratio_thresh:
            if x2 - x1 > y2 - y1:
                avg_x = (x1 + x2) / 2
                # mid left point
                avg_x_left = (x1 + avg_x) / 2
                m = avg_x_left - x1
                t = x1 / m
                mid_x_left = m * (np.random.rand() + t)

                # mid right point
                avg_x_right = (x2 + avg_x) / 2
                m = x2 - avg_x_right
                t = avg_x_right / m
                mid_x_right = m * (np.random.rand() + t)

                mid_y = (y1 + y2) / 2
                return mid_x_left, mid_y, mid_x_right, mid_y, big_aspect_object
            else:
                avg_y = (y1 + y2) / 2
                # mid up point
                avg_y_up = (y1 + avg_y) / 2
                m = avg_y_up - y1
                t = y1 / m
                mid_y_up = m * (np.random.rand() + t)
                # mid down point
                avg_y_down = (y2 + avg_y) / 2
                m = y2 - avg_y_down
                t = avg_y_down / m
                mid_y_down = m * (np.random.rand() + t)

                mid_x = (x1 + x2) / 2
                return mid_x, mid_y_up, mid_x, mid_y_down, big_aspect_object
        # if object is  median, sample one point ,and alpha threshold should be small
        if area_ratio < median_object_thresh:
            avg_x = (x1 + x2) / 2
            x1 = (x1 + avg_x) / 2
            x2 = (x2 + avg_x) / 2
            m = x2 - x1
            t = x1 / m
            mid_x = m * (np.random.rand() + t)

            avg_y = (y1 + y2) / 2
            y1 = (y1 + avg_y) / 2
            y2 = (y2 + avg_y) / 2
            m = y2 - y1
            t = y1 / m
            mid_y = m * (np.random.rand() + t)

            # if mid_x > (x1 + x2) / 2:
            #     print(x1, x2, mid_x)
            return mid_x, mid_y, 0, 0, median_object
        else:  # if object is too big ,sample 2 points along the diagonal
            avg_x = (x1 + x2) / 2
            avg_y = (y1 + y2) / 2
            # mid up-left point
            avg_x_left = (x1 + avg_x) / 2
            m = avg_x_left - x1
            t = x1 / m
            mid_x_left = m * (np.random.rand() + t)

            avg_y_up = (y1 + avg_y) / 2
            m = avg_y_up - y1
            t = y1 / m
            mid_y_up = m * (np.random.rand() + t)

            # mid down-right point
            avg_x_right = (x2 + avg_x) / 2
            m = x2 - avg_x_right
            t = avg_x_right / m
            mid_x_right = m * (np.random.rand() + t)

            avg_y_down = (y2 + avg_y) / 2
            m = y2 - avg_y_down
            t = avg_y_down / m
            mid_y_down = m * (np.random.rand() + t)

            return mid_x_left, mid_y_up, mid_x_right, mid_y_down, large_object

def generate_simulated_center(x,y,small_object):

    simulated_x,simulated_y = 0,0
    if small_object is True:
        simulated_x = np.random.randint(low=x - SMALL_OBJECT_TO_BBX_CENETER, high=x+SMALL_OBJECT_TO_BBX_CENETER)
        simulated_y = np.random.randint(low=y-SMALL_OBJECT_TO_BBX_CENETER,high=y+SMALL_OBJECT_TO_BBX_CENETER)
    else:
        simulated_x = np.random.randint(low=x-BIG_OBJECT_TO_BBX_CENTER,high=x+BIG_OBJECT_TO_BBX_CENTER)
        simulated_y = np.random.randint(low=y-BIG_OBJECT_TO_BBX_CENTER,high=y+BIG_OBJECT_TO_BBX_CENTER)
    return simulated_x,simulated_y
if __name__ == '__main__':
    d = datasets.pascal_voc_gt_mid('trainval', '2007')
    # res = d.roidb
    # from IPython import embed; embed()
    ss_box_list = d.load_ss_roidb()
    # 2 random coords to get more rubust results
    # object_mid_2_coords = d.gt_mid_coords_roidb('random_mid_2_coords_roidb')
    # 1 coods which is the center of one object(the center of bbox)
    object_mid_coords = d.gt_mid_coords_roidb('mid_coords_roidb')
    object_simulated_center = d.gt_mid_coords_roidb('simulated_center')
    # cache_file = os.path.join(d.cache_path, d.name + '_pred_not_diff_gt_roidb_overlap.pkl')
    cache_file = os.path.join(d.cache_path, d.name + '_gt_mid_coords_roidb.pkl')
    print(cache_file)
    total_boxes = 0
    true_postive_boxes = 0
    if os.path.exists(cache_file):
        with open(cache_file, 'rb') as fid:
            pred_roidb = cPickle.load(fid)
            # print(len(pred_roidb))
            for pred in pred_roidb:
                # print(pred,pred['image_name'],pred['simlu_centers'])
                # pred = pred_roidb[5]
                total_boxes += len(pred['boxes'])
                print(total_boxes)
                # break
                # print(len(pred),len(pred['corLoc']),pred['corLoc'].astype(np.int8))#, pred_roidb[0], pred['gt_overlaps'].toarray())
                # for corloc in pred['corLoc'].astype(np.int8):
                #     if corloc[0] == 1:
                #         true_postive_boxes +=1
                    # print(true_postive_boxes)
    # print('corLoc:',1.0*true_postive_boxes/total_boxes,true_postive_boxes,total_boxes)


    # pred_results = []
    #
    # box_number_count = 0
    # filter_number_count = 0
    # for index in range(len(object_mid_coords)):
    #     im_name = os.path.join('/home/lzhang/fast-rcnn/data/VOCdevkit2007/'
    #                            'VOC2007/JPEGImages', object_mid_coords[index]['index'] + '.jpg')
    #     im = cv2.imread(im_name)
    #     print('img_index:',index,'img_name:',object_mid_coords[index]['index'])
        # print(ss_box_list[i],len(ss_box_list[i]))
        # print(im.shape, im.shape[0] / 10 + 1)
        # print(ss_box_list[i].shape,ss_box_list[i][0][1])
        #     if im.shape[1] % 10 is not 0:
        #         d.get_grid_bbox(im)
        #         print('___', im.shape)
        #         break
        #     print(im.shape)
        # pred_result = d.vis_detections(im, object_simulated_center,object_mid_coords, ss_box_list[index],index)
        # pred_result['image_name'] = object_mid_2_coords[index]['index']
        # pred_results.append(pred_result)
        # print(pred_result)
        # print(len(ss_box_list[index]),pred_result['filter_length'])
    #     box_number_count += len(ss_box_list)
    #     filter_number_count += pred_result['filter_length']
    # print('total_boxes:',box_number_count,'filter_boxes:',filter_number_count)
    #     pred_results.append(pred_result)
    # with open(cache_file, 'wb') as fid:
    #     cPickle.dump(pred_results, fid, cPickle.HIGHEST_PROTOCOL)
    # print 'wrote gt roidb to {}'.format(cache_file)

