#!/bin/bash


export PYTHONPATH=$(pwd)

PY=python
SOURCE=scripts/ensemble.py
DEVICE=cuda
DEVICES=0
WORKERS=12
SEED=0

DATASET=voc12
DATA_DIR=/home/ldavid/workspace/datasets/voc/VOCdevkit/VOC2012/
DOMAIN=train


run_ensemble () {
    CUDA_VISIBLE_DEVICES=$DEVICES           \
    $PY $SOURCE                             \
    --seed           $SEED                  \
    --tag            $TAG                   \
    --merge          $MERGE                 \
    --weights_path   "$WEIGHTS_PATH"        \
    --dataset        $DATASET               \
    --domain         $DOMAIN                \
    --data_dir       $DATA_DIR              \
    --num_workers    $WORKERS               \
    --experiments    $EXPERIMENTS
}


TAG=ensemble/ra-oc-p-avg
EXPERIMENTS="vanilla/resnest269@randaug@train@scale=0.5,1.0,1.5,2.0 literature/rn38d-occse puzzle/resnest269@puzzlerep2@train@scale=0.5,1.0,1.5,2.0"
MERGE=avg
WEIGHTS_PATH=""
# run_ensemble

# TAG=ensemble/ra-oc-l2g-p-pnoc-avg
# EXPERIMENTS="vanilla/resnest269@randaug@train@scale=0.5,1.0,1.5,2.0 literature/rn38d-occse literature/l2g puzzle/resnest269@puzzlerep2@train@scale=0.5,1.0,1.5,2.0 pnoc/voc12-rs269-pnoc-ls0.1-ow0.0-1.0-1.0-cams-0.2-octis1-amp@rs269ra-r3@train@scale=0.5,1.0,1.5,2.0"
# MERGE=avg
# run_ensemble

TAG=ensemble/ra-oc-p-poc-pnoc-highest
EXPERIMENTS="vanilla/resnest269@randaug@train@scale=0.5,1.0,1.5,2.0 literature/rn38d-occse puzzle/resnest269@puzzlerep2@train@scale=0.5,1.0,1.5,2.0 poc/voc12-rs269-poc-ls0.1@rs269ra-r3@train@scale=0.5,1.0,1.5,2.0 pnoc/voc12-rs269-pnoc-ls0.1-ow0.0-1.0-1.0-cams-0.2-octis1-amp@rs269ra-r3@train@scale=0.5,1.0,1.5,2.0"
MERGE=highest  # weighted, ranked
WEIGHTS_PATH=experiments/logs/ious.csv
run_ensemble

# WEIGHTS=puzzle/ResNeSt269@Puzzle@optimal
# WEIGHTS=puzzle/resnest269@puzzlerep
# WEIGHTS=puzzle/resnest269@puzzlerep2
# WEIGHTS=poc/ResNeSt269@PuzzleOc


## =========================================
## MS COCO Dataset
## =========================================


# DATASET=coco14
# DATA_DIR=/home/ldavid/workspace/datasets/coco14/
# DOMAIN=train2014

# WEIGHTS=pnoc/coco14-rs269-pnoc-b16-a2-ls0.1-ow0.0-1.0-1.0-c0.2-is1@rs269ra-r3 pnoc/coco14-rs269-pnoc-b16-a2-lr0.05-ls0-ow0.0-1.0-1.0-c0.2-is1@rs269ra-r1
# TAG=$WEIGHTS
# # DOMAIN=val2014
# # run_ensemble
# run_ensemble
