#!/usr/bin/env bash
set -x

T=`date +%m%d%H%M`
JOB_NAME='uniad_eval'


# -------------------------------------------------- #
# Usually you only need to customize these variables #
PARTITION=$1                                         #
CFG=$2                                               #    
CKPT=$3                                              #    
GPUS=${4:-8}                                         #
# -------------------------------------------------- #

PY_ARGS=${@:5}
GPUS_PER_NODE=$(($4<8?$4:8))
CPUS_PER_TASK=${CPUS_PER_TASK:-5}

SRUN_ARGS=${SRUN_ARGS:-""}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \

WORK_DIR=$(echo ${CFG%.*} | sed -e "s/configs/work_dirs/g")/
# Intermediate files and logs will be saved to UniAD/projects/work_dirs/

if [ ! -d ${WORK_DIR}logs ]; then
    mkdir -p ${WORK_DIR}logs
fi
srun -p ${PARTITION} \
    --job-name=${JOB_NAME} \
    --gres=gpu:${GPUS_PER_NODE} \
    --ntasks=${GPUS} \
    --ntasks-per-node=${GPUS_PER_NODE} \
    --cpus-per-task=${CPUS_PER_TASK} \
    --kill-on-bad-exit=1 \
    ${SRUN_ARGS} \
    python -W ignore -u tools/test.py ${CFG} ${CKPT} \
    --eval bbox \
    --show-dir ${WORK_DIR} \
    --launcher="slurm" ${PY_ARGS} \
    2>&1 | tee ${WORK_DIR}logs/eval.$T