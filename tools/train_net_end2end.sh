#!/usr/bin/env bash
python ./tools/train_net.py \
        --gpu 0 \
        --solver models/pascal_voc/VGG16/faster_rcnn_end2end/solver.prototxt \
	    --weights data/imagenet_models/VGG16.v2.caffemodel \
	    --imdb voc_2007_trainval \
	    --iters 200000 \
	    --cfg experiments/cfgs/faster_rcnn_end2end.yml \
	    --set EXP_DIR classfi RNG_SEED 42 TRAIN.SCALES "[400,500,600,700]"