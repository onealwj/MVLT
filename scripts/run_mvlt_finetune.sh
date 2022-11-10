#!/usr/bin/env bash

set -x

METHOD=mvlt_vit_base_patch16
DATA_PATH="./data/training"
DATA_PATH_VAL="./data/evaluation"
CHECKPOINT=${1}
LR=2.5e-6
WD=0.05
BS=128
NUM_ITER=20000
WARM_IT=8000
PY_ARGS=${@:1}
PORT=${PORT:-23456}
NPROC=${NPROC:-8}
PYTHON=${PYTHON:-"python"}

${PYTHON} -u -m torch.distributed.launch --master_port ${PORT} --nproc_per_node=${NPROC} \
    main_mvlt_finetune.py \
    --imgH 112 \
    --imgW 448 \
    --accum_iter 1 \
    --batch_size ${BS} \
    --model ${METHOD} \
    --num_iter ${NUM_ITER} \
    --warmup_iters ${WARM_IT} \
    --blr ${LR} \
    --min_lr 5e-7 \
    --weight_decay ${WD} \
    --data_path ${DATA_PATH} \
    --data_path_val ${DATA_PATH_VAL} \
    --finetune ${CHECKPOINT} \
    --iter_correct 5 \
    --clip_grad 2.0 \
    --abinet_augment \
