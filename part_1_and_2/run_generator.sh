#!/bin/bash
path_to_train=$1
# path_to_test_query=$2
# path_to_output_file=$3
path_to_sample_images=$2
path_to_gen=$3
path_to_target=$4
# output_joint=$7

python final_part_1.py --train_path "$path_to_train" --sample_images "$path_to_sample_images" --gen_path "$path_to_gen" --target_path "$path_to_target"
python -m pytorch_fid fid_0 fid_1 --batch-size 128 --device cuda
