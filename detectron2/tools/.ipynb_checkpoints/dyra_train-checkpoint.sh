#!/bin/bash

model_type="retinanet_R_101_FPN_2x"
config_file="../configs/DyRA/${model_type}.yaml"

export DETECTRON2_DATASETS="/home/data"
~/anaconda3/envs/py_37/bin/python3.7 train_net.py --config-file $config_file --num-gpus 4 # --resume
