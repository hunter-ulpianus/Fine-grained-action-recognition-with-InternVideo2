#!/bin/bash

find_free_port() {
    local port
    for port in $(seq 12000 31999); do
        (echo > /dev/tcp/127.0.0.1/$port) &>/dev/null
        if [ $? -ne 0 ]; then
            echo $port
            return
        fi
    done
}

export MASTER_PORT=$(find_free_port)
export OMP_NUM_THREADS=1
export PYTHONPATH="/media/sdc/fe/planB/InternVideo/InternVideo2/multi_modality"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export python="/media/sdc/fe/anaconda3/envs/internvideo2_backup/bin/python3.8"
which_python=$(which python)
echo "which python: ${which_python}"
echo "PYTHONPATH: ${PYTHONPATH}"

JOB_NAME='1B'
OUTPUT_DIR="/media/sdc/fe/ckpt_planb_1B"
LOG_DIR="/media/sdc/fe/planb_1B_log"

# Number of GPUs to use
NUM_GPUS=8


torchrun \
    --nnodes 1 \
    --nproc_per_node=${NUM_GPUS} \
    --node_rank=0 \
    --master_addr="127.0.0.1" \
    --master_port=${MASTER_PORT} \
    /media/sdc/fe/planB/InternVideo/InternVideo2/multi_modality/tasks_clip/pretrain.py \
    $(dirname $0)/config.py \
    output_dir ${OUTPUT_DIR}
