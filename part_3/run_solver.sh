#!/bin/bash
path_to_train=$1
path_to_test_query=$2
path_to_output_file=$3
path_to_sample_images=$4
path_to_gen_joint=$5
path_to_target_joint=$6
output_joint=$7

python part3.1.py --train_path "$path_to_train"
python part3.1.1.py --train_path "$path_to_train" --test_query "$path_to_test_query"
python part3.2.py --train_path "$path_to_train" --test_query "$path_to_test_query" --sample_images "$path_to_sample_images"
python part3.3.py --test_query "$path_to_test_query" --output_csv "$output_joint"
python part3.4.py --gen_path "$path_to_gen_joint" --target_path "$path_to_target_joint"
python -m pytorch_fid fid_0 fid_1 --batch-size 128 --device cuda