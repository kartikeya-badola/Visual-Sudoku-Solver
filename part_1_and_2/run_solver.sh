#!/bin/bash
path_to_train=$1
path_to_test_query=$2
path_to_sample_images=$3
output_csv=$4

python try_repeat.py
python part2.py --train_path "$path_to_train" --test_query "$path_to_test_query"
python part2_2.py --test_query "$path_to_test_query" --output_csv "$output_csv"