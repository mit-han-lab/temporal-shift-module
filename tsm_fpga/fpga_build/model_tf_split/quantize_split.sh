#!/bin/bash
set -e

export DECENT_DEBUG=3

# Constants
num_splits=11
calib_iter=50

# Path to output directory of TF
base_path="$1"
base_options="--calib_iter $calib_iter --input_fn calib_input_split.input_fn"

num_split_dirs=$(ls "$base_path" | wc -l)

if [[ num_splits -ne $num_split_dirs ]]; then
    echo "Number of outputs split directories from: \n\
        '$base_path' ($num_split_dirs)\n not equal to coded num_splits ($num_splits)"
fi

export CALIB_BASE_PATH="$base_path"

if [[ $# -eq 0 ]]; then
    echo "Missing arg: Provide path to base split model dir"
    exit 1
fi

for ((i=0;i<num_splits;i++)); do
    printf "\n================ Quantizing split # $i ====================\n"
    model_dir="$base_path/model_tf_split_$i"
    config=$(<"$model_dir/quantize_info.txt")
    export CALIB_MODEL_SPLIT=$i

    tee_append=""
    if [[ $i -ne 0 ]]; then
        tee_append="-a"
    fi

    vai_q_tensorflow quantize --output_dir "quantize_results/quantize_results_$i" $base_options --input_frozen_graph "$model_dir/model_tf_split_$i.pb" \
        $(echo $config) 2>&1 | tee $tee_append quantize_log.txt
done

