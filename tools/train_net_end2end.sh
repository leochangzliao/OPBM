#!/usr/bin/env bash
LOG=data/cache/log/train-`date +%Y-%m-%d-%H-%M-%S`.log
#cuda-gdb --args
python ./tools/train_net.py \
        --gpu 0 \
        --solver models/pascal_voc/VGG16/faster_rcnn_end2end/solver.prototxt \
	    --weights data/imagenet_models/VGG16.v2.caffemodel \
	    --imdb voc_2007_trainval \
	    --iters 300000 \
	    --cfg experiments/cfgs/faster_rcnn_end2end.yml \
	    --set EXP_DIR classfi_MAX_Pool RNG_SEED 43 TRAIN.SCALES "[400,500,600,700]" 2>&1 | tee $LOG
