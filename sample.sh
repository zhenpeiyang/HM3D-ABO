#!/bin/bash
conda activate fvor
BASE_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
OUTPUTS_DIR=$BASE_DIR/data/camera_pose_configs

n_gpus=$(nvidia-smi --list-gpus | wc -l)
n_jobs_per_gpu=2
n_jobs=$(($n_gpus*n_jobs_per_gpu))

cd third_party/habitat-sim/ && seq 0 $((n_jobs-1)) | parallel --ungroup python ../../sample.py --camera_mode v2 --height=120 --width=160 --remove_margin --n_proc $n_jobs --process_id {} --num_gpu $n_gpus --assets_dir $BASE_DIR/data --outputs_dir $OUTPUTS_DIR --num_scene_sample_per_object 3 && cd ../../

#debug
#cd third_party/habitat-sim && python ../../sample.py --camera_mode v2 --height=120 --width=160 --remove_margin --n_proc 1 --process_id 0 --num_gpu 1 --assets_dir $BASE_DIR/data --outputs_dir $OUTPUTS_DIR --num_scene_sample_per_object 3
