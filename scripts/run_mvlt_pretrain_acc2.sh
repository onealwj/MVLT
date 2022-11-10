#!/usr/bin/env bash

set -x

METHOD=mvlt_vit_base_patch16
DATA_PATH="./data/training"
LR=1.5e-4
WD=0.05
BS=256
NUM_ITER=240000
MASK_RATIO=0.75
WARM_IT=16000
PY_ARGS=${@:1}
PORT=${PORT:-23456}
NPROC=${NPROC:-8}
PYTHON=${PYTHON:-"python"}

${PYTHON} -u -m torch.distributed.launch --master_port ${PORT} --nproc_per_node=${NPROC} \
    main_mvlt_pretrain.py \
    --batch_size ${BS} \
    --accum_iter 2 \
    --model ${METHOD} \
    --mask_ratio ${MASK_RATIO} \
    --num_iter ${NUM_ITER} \
    --warmup_iters ${WARM_IT} \
    --blr ${LR} \
    --weight_decay ${WD} \
    --data_path ${DATA_PATH} \
    --random_crop \
    --exp_semantic \
    --imp_semantic \
