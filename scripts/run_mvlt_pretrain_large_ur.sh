#!/usr/bin/env bash

set -x

METHOD=mvlt_vit_large_patch16
DATA_PATH="./data/training"
LR=1.25e-4
WD=0.05
BS=192
NUM_ITER=480000
MASK_RATIO=0.75
WARM_IT=32000
PY_ARGS=${@:1}
PORT=${PORT:-23456}
NPROC=${NPROC:-8}
PYTHON=${PYTHON:-"python"}

${PYTHON} -u -m torch.distributed.launch --master_port ${PORT} --nproc_per_node=${NPROC} \
    main_mvlt_pretrain.py \
    --select_data RealUnlabel-RealLabel-MJ-ST \
    --batch_ratio 0.17-0.17-0.33-0.33 \
    --accum_iter 4 \
    --batch_size ${BS} \
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
