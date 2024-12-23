#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=48
#SBATCH -p sequana_gpu_shared
#SBATCH -J sal-ccamh
#SBATCH -o /scratch/lerdl/lucas.david/logs/%j-sal-ccamh.out
#SBATCH --time=4:00:00

# Copyright 2023 Lucas Oliveira David
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#
# Saliency Detection with C²AM-H.
#

if [[ "`hostname`" == "sdumont"* ]]; then
  ENV=sdumont
  WORK_DIR=$SCRATCH/pnoc
else
  ENV=local
  WORK_DIR=$HOME/workspace/repos/research/wsss/pnoc
fi

# Dataset
DATASET=voc12  # Pascal VOC 2012
# DATASET=coco14  # MS COCO 2014
# DATASET=deepglobe # DeepGlobe Land Cover Classification

. $WORK_DIR/runners/config/env.sh
. $WORK_DIR/runners/config/dataset.sh

cd $WORK_DIR
export PYTHONPATH=$(pwd)

IMAGE_SIZE=448
EPOCHS=10
BATCH_SIZE=64
ACCUMULATE_STEPS=2
LABELSMOOTHING=0.1

MIXED_PRECISION=true
PERFORM_VALIDATION=true

ARCHITECTURE=resnest269
ARCH=rs269
DILATED=false
TRAINABLE_STEM=true
MODE=normal

ALPHA=0.25
HINT_W=1.0
LR=0.001

CAMS_MODE=npy
FG_T=0.4
BG_T=0.05

INF_FG_T=0.2
CRF_T=10
CRF_GT_PROB=0.7

# Evaluation
EVAL_MIN_TH=0.40
EVAL_MAX_TH=0.61
EVAL_MODE=saliency
EVAL_FILE_MODE=npy


ccam_training() {
  echo "=================================================================="
  echo "[train $TAG] started at $(date +'%Y-%m-%d %H:%M:%S')."
  echo "=================================================================="

  CUDA_VISIBLE_DEVICES=0,1,2,3 \
    WANDB_TAGS="$DATASET,$ARCH,ccam" \
    WANDB_TAGS="ccam,amp,$DATASET,$ARCH,b:$BATCH_SIZE,ac:$ACCUMULATE_STEPS,lr:$LR" \
    $PY scripts/ccam/train.py \
    --tag $TAG \
    --alpha $ALPHA \
    --max_epoch $EPOCHS \
    --batch_size $BATCH_SIZE \
    --lr $LR \
    --accumulate_steps $ACCUMULATE_STEPS \
    --mixed_precision $MIXED_PRECISION \
    --num_workers $WORKERS_TRAIN \
    --architecture $ARCHITECTURE \
    --dilated $DILATED \
    --mode $MODE \
    --weights $WEIGHTS \
    --trainable-stem $TRAINABLE_STEM \
    --image_size $IMAGE_SIZE \
    --data_dir $DATA_DIR
}

ccamh_training() {
  echo "=================================================================="
  echo "[train $CCAMH_TAG] started at $(date +'%Y-%m-%d %H:%M:%S')."
  echo "=================================================================="

  WANDB_TAGS="ccamh,amp,$DATASET,$ARCH,b:$BATCH_SIZE,ac:$ACCUMULATE_STEPS,fg:$FG_T,ls:$LABELSMOOTHING,lr:$LR" \
    CUDA_VISIBLE_DEVICES=$DEVICES $PY scripts/ccam/train_hints.py \
    --tag $CCAMH_TAG \
    --alpha $ALPHA \
    --hint_w $HINT_W \
    --max_epoch $EPOCHS \
    --batch_size $BATCH_SIZE \
    --lr $LR \
    --label_smoothing $LABELSMOOTHING \
    --accumulate_steps $ACCUMULATE_STEPS \
    --mixed_precision $MIXED_PRECISION \
    --validate $PERFORM_VALIDATION \
    --validate_max_steps $VALIDATE_MAX_STEPS \
    --num_workers $WORKERS_TRAIN \
    --architecture $ARCHITECTURE \
    --dilated $DILATED \
    --mode $MODE \
    --trainable-stem $TRAINABLE_STEM \
    --image_size $IMAGE_SIZE \
    --cams_dir $CAMS_DIR \
    --fg_threshold $FG_T \
    --bg_threshold $BG_T \
    --cams_mode $CAMS_MODE \
    --dataset $DATASET \
    --data_dir $DATA_DIR
}

ccamh_inference() {
  echo "=================================================================="
  echo "[inference $CCAMH_TAG] started at $(date +'%Y-%m-%d %H:%M:%S')."
  echo "=================================================================="

  WEIGHTS=imagenet
  PRETRAINED=./experiments/models/$CCAMH_TAG.pth

  CUDA_VISIBLE_DEVICES=$DEVICES $PY scripts/ccam/inference.py \
    --tag $CCAMH_TAG \
    --dataset $DATASET \
    --domain $DOMAIN \
    --architecture $ARCHITECTURE \
    --dilated $DILATED \
    --mode $MODE \
    --weights $WEIGHTS \
    --trainable-stem $TRAINABLE_STEM \
    --pretrained $PRETRAINED \
    --data_dir $DATA_DIR
}

ccamh_pseudo_masks_crf() {
  echo "=================================================================="
  echo "[pseudo masks dCRF $CCAMH_TAG] started at $(date +'%Y-%m-%d %H:%M:%S')."
  echo "=================================================================="
  $PY scripts/ccam/inference_crf.py \
    --experiment_name $CCAMH_TAG@train@scale=0.5,1.0,1.5,2.0 \
    --dataset $DATASET \
    --domain $DOMAIN \
    --threshold $INF_FG_T \
    --crf_t $CRF_T \
    --crf_gt_prob $CRF_GT_PROB \
    --data_dir $DATA_DIR
}

poolnet_training() {
  cd $WORK_DIR/poolnet

  CUDA_VISIBLE_DEVICES=0 $PY main_custom.py \
    --arch "resnet" \
    --mode "train" \
    --dataset $DATASET \
    --train_root $DATA_DIR \
    --train_list $DOMAIN \
    --pseudo_root $SAL_PRIORS_DIR

  cd $WORK_DIR
}

poolnet_inference() {
  cd $WORK_DIR/poolnet

  CUDA_VISIBLE_DEVICES=0 $PY main_custom.py \
    --arch "resnet" \
    --mode "test" \
    --model $PN_CKPT \
    --dataset $DATASET \
    --train_list $DOMAIN \
    --train_root $DATA_DIR \
    --pseudo_root $SAL_PRIORS_DIR \
    --sal_folder ./results/$PN_TAG

  cd $WORK_DIR
}

evaluate_saliency_detection() {
  WANDB_TAGS="$DATASET,domain:$DOMAIN,ccamh" \
    $PY scripts/ccam/evaluate.py \
    --experiment_name "$TAG" \
    --dataset $DATASET \
    --domain $DOMAIN \
    --min_th $EVAL_MIN_TH \
    --max_th $EVAL_MAX_TH \
    --mode $EVAL_FILE_MODE \
    --eval_mode $EVAL_MODE \
    --crf_t $CRF_T \
    --crf_gt_prob $CRF_GT_PROB \
    --data_dir $DATA_DIR \
    --num_workers $WORKERS_INFER;
}

## ================================================
## Pascal VoC 2012
##
ARCHITECTURE=resnest101
ARCH=rs101
CAMS_DIR=experiments/predictions/csrm/voc12-rs101-lr0.007-m0.9-b32-classmix-ls-sdefault-u1-c1-r1@train/cams
PRIORS_TAG=rs101csrm@rs101p

# ARCHITECTURE=resnest269
# ARCH=rs269
# CAMS_DIR=experiments/predictions/csrm/voc12-rs269-lr0.1-m0-b16-classmix-ls-sdefault-bg0.05-fg0.40-u1-c1-r1@train/cams
# PRIORS_TAG=rs269csrm@rs269pnoc

FG_T=0.4
INF_FG_T=0.5

## ================================================
## MS COCO 2014
##
# FG_T=0.3
# LR=0.0005

# CAMS_DIR=experiments/predictions/csrm/coco14-640-rs269-lr0.007-m0.9-b32-colorjitter_classmix-default-bg0.05-fg0.35-u1-c1@rs269pnoc-r1@train/pseudos-t0.4-c10
# CAMS_MODE=png
# PRIORS_TAG=rs269csrm@rs269pnoc-r1
## ================================================

CCAMH_TAG=saliency/$DATASET-ccamh-$ARCH-fg$FG_T-lr$LR-b$BATCH_SIZE@$PRIORS_TAG

PN_TAG=$DATASET-pn@ccamh-$ARCH-fg$FG_T@$PRIORS_TAG
PN_ID=run-0
PN_CKPT=$WORK_DIR/poolnet/results/$PN_ID/models/epoch_9.pth

ccamh_training
ccamh_inference
ccamh_pseudo_masks_crf

## PoolNet Training and Inference
## ==============================
SAL_PRIORS_DIR=$WORK_DIR/experiments/predictions/$CCAMH_TAG@train@scale=0.5,1.0,1.5,2.0@t=$INF_FG_T@crf=$CRF_T

# poolnet_training
DOMAIN=$DOMAIN_TRAIN poolnet_inference
DOMAIN=$DOMAIN_VALID_SEG poolnet_inference

cp $PN_CKPT $WORK_DIR/experiments/models/saliency/$PN_TAG.pth

mv $WORK_DIR/poolnet/results/$PN_TAG $WORK_DIR/experiments/predictions/saliency/

## Cleanup model checkpoints so next run doesn't change the PN_ID.
## If this line is commented, you must manually remove this directory
## or set PN_ID to the next integer before running this script.
# rm -rf $WORK_DIR/poolnet/results/$PN_ID

## Evaluation
## ==============================

CRF_T=10 TAG=$CCAMH_TAG@train@scale=0.5,1.0,1.5,2.0 DOMAIN=$DOMAIN_VALID evaluate_saliency_detection
CRF_T=0  TAG=saliency/$PN_TAG DOMAIN=$DOMAIN_VALID evaluate_saliency_detection
