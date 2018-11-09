#!/usr/bin/env bash
python ./tools/test_net.py \
--gpu 0 \
--def models/pascal_voc/VGG16/faster_rcnn_end2end/test.prototxt \
--net output/classfi2/voc_2007_trainval/vgg16_faster_rcnn_simu_rpn_iter_80000.caffemodel \
--imdb voc_2007_test \
--cfg experiments/cfgs/faster_rcnn_end2end.yml
