#!/bin/bash
BASE_DIR='./'
DATASET='HM3D_ABO'
TEMP_DIR=$BASE_DIR/data/$DATASET/temp
OUTPUT_DIR=$BASE_DIR/data/$DATASET/scenes
CONFIG_DIR=$BASE_DIR/data/camera_pose_configs
mkdir -p $TEMP_DIR
mkdir -p $OUTPUT_DIR
export n_gpus=$(nvidia-smi --list-gpus | wc -l)
n_jobs_per_gpu=16
n_jobs=$((n_gpus*n_jobs_per_gpu))

## Do the actual rendering
seq 0 $((n_jobs-1))|parallel -j $(($n_gpus*4)) --ungroup CUDA_VISIBLE_DEVICES='$(({}%$n_gpus))' ./third_party/blender-2.93.1-linux-x64/blender --background --python my_blender.py --  --output_folder $TEMP_DIR --format OPEN_EXR --engine CYCLES  --n_proc $n_jobs --process_id {} --gpu --lens 30

## Make the final dataset.
python create_dataset.py --temp_folder $TEMP_DIR --output_folder $OUTPUT_DIR --configs_folder $CONFIG_DIR


## Delete temp folder
rm -rf $TEMP_DIR
