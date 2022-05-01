#!/bin/bash
python=/opt/conda/envs/habitat/bin/python
BASE_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
DATASET='HM3D_ABO'
OUTPUT_DIR=$BASE_DIR/data/$DATASET/abo_assets
TEMP_DIR=$BASE_DIR/data/abo_denseview
mkdir -p $TEMP_DIR

## Render dense depth images surrounding the object
export n_gpus=$(nvidia-smi --list-gpus | wc -l)
n_jobs=$(($n_gpus*2))

seq 0 $((n_jobs-1))|parallel --ungroup CUDA_VISIBLE_DEVICES='$(({}%n_gpus))' ./third_party/blender-2.93.1-linux-x64/blender --background --python my_blender_abo.py --  --assets_folder $BASE_DIR/data --output_folder $TEMP_DIR  --format OPEN_EXR --engine CYCLES  --n_proc $n_jobs --process_id {} --gpu --views 100 --height 512 --width 512

## Fuse into water-tight mesh
python tsdf_fusion.py --depth_folder $TEMP_DIR

## Sample points and compute SDF
python sample_mesh.py --n_proc 1 --resize --input_folder $TEMP_DIR --float16 --output_folder $OUTPUT_DIR

rm -rf $TEMP_DIR

