#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=48
#SBATCH -p sequana_gpu_shared
#SBATCH -J sal-ccamh
#SBATCH -o /scratch/lerdl/lucas.david/logs/sal-%j.out
#SBATCH --time=24:00:00

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

ENV=sdumont
WORK_DIR=$SCRATCH/PuzzleCAM
# ENV=local
# WORK_DIR=/home/ldavid/workspace/repos/research/pnoc

# Dataset
# DATASET=voc12  # Pascal VOC 2012
# DATASET=coco14  # MS COCO 2014
DATASET=deepglobe # DeepGlobe Land Cover Classification

. $WORK_DIR/runners/config/env.sh
. $WORK_DIR/runners/config/dataset.sh

cd $WORK_DIR
export PYTHONPATH=$(pwd)

IMAGE_SIZE=448
EPOCHS=10
BATCH_SIZE=64
ACCUMULATE_STEPS=2
MIXED_PRECISION=true
LABELSMOOTHING=0.1

ARCHITECTURE=resnest269
ARCH=rs269
DILATED=false
TRAINABLE_STEM=true
MODE=normal
S4_OUT_FEATURES=1024

ALPHA=0.25
HINT_W=1.0
LR=0.001

FG_T=0.3
# BG_T=0.1

INF_FG_T=0.2
CRF_T=10
CRF_GT_PROB=0.7

ccam_training() {
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
    --stage4_out_features $S4_OUT_FEATURES \
    --dilated $DILATED \
    --mode $MODE \
    --weights $WEIGHTS \
    --trainable-stem $TRAINABLE_STEM \
    --image_size $IMAGE_SIZE \
    --data_dir $DATA_DIR
}

ccamh_training() {
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
    --num_workers $WORKERS_TRAIN \
    --architecture $ARCHITECTURE \
    --stage4_out_features $S4_OUT_FEATURES \
    --dilated $DILATED \
    --mode $MODE \
    --trainable-stem $TRAINABLE_STEM \
    --image_size $IMAGE_SIZE \
    --cams_dir $CAMS_DIR \
    --fg_threshold $FG_T \
    --dataset $DATASET \
    --data_dir $DATA_DIR
  # --bg_threshold    $BG_T                \
}

ccamh_inference() {
  WEIGHTS=imagenet
  PRETRAINED=./experiments/models/$CCAMH_TAG.pth

  CUDA_VISIBLE_DEVICES=$DEVICES $PY scripts/ccam/inference.py \
    --tag $CCAMH_TAG \
    --dataset $DATASET \
    --domain $DOMAIN \
    --architecture $ARCHITECTURE \
    --dilated $DILATED \
    --stage4_out_features $S4_OUT_FEATURES \
    --mode $MODE \
    --weights $WEIGHTS \
    --trainable-stem $TRAINABLE_STEM \
    --pretrained $PRETRAINED \
    --data_dir $DATA_DIR
}

ccamh_pseudo_masks_crf() {
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
  cd $PRJ_DIR/poolnet

  CUDA_VISIBLE_DEVICES=0 $PY main_custom.py \
    --arch "resnet" \
    --mode "train" \
    --dataset $DATASET \
    --train_root $DATA_DIR \
    --train_list $DOMAIN \
    --pseudo_root $SAL_PRIORS_DIR

  cd $PRJ_DIR
}

poolnet_inference() {
  cd $PRJ_DIR/poolnet

  CUDA_VISIBLE_DEVICES=0 $PY main_custom.py \
    --arch "resnet" \
    --mode "test" \
    --model $PN_CKPT \
    --dataset $DATASET \
    --train_list $DOMAIN \
    --train_root $DATA_DIR \
    --pseudo_root $SAL_PRIORS_DIR \
    --sal_folder ./results/$PN_TAG

  cd $PRJ_DIR
}

evaluate_saliency_detection() {
  WANDB_TAGS="$DATASET,domain:$DOMAIN,ccamh,pn" \
    $PY scripts/ccam/evaluate.py \
    --experiment_name "$TAG" \
    --pred_dir $PRED_DIR \
    --dataset $DATASET \
    --domain $DOMAIN \
    --min_th 0.05 \
    --max_th 0.81 \
    --mode npy \
    --eval_mode saliency \
    --crf_t $CRF_T \
    --crf_gt_prob $CRF_GT_PROB \
    --data_dir $DATA_DIR \
    --num_workers $WORKERS_INFER
}

## Pascal VoC 2012
##
# FG_T=0.4
# CAMS_DIR=experiments/predictions/pnoc/voc12-rs269-pnoc-ls0.1-ow0.0-1.0-1.0-cams-0.2-octis1-amp@rs269ra-r3@train@scale=0.5,1.0,1.5,2.0
# CCAMH_TAG=saliency/$DATASET-ccamh-$ARCH@rw269pnoc@rs269@b$BATCH_SIZE-fg$FG_T-lr$LR-b$BATCH_SIZE
##
## ================================================

## MS COCO 2014
##
# FG_T=0.3
# LR=0.0005
# CAMS_DIR=experiments/predictions/pnoc/coco14-rs269-pnoc-b16-a2-lr0.05-ls0-ow0.0-1.0-1.0-c0.2-is1@rs269ra-r1@train@scale=0.5,1.0,1.5,2.0
# CCAMH_TAG=saliency/$DATASET-ccamh-$ARCH@rs269pnoc-lr0.05@rs269@b$BATCH_SIZE-fg$FG_T-lr$LR-b$BATCH_SIZE
##
## ================================================

## Ensemble
##
FG_T=0.3

# ENS=ra-oc-p-poc-pnoc-avg
# CAMS_DIR=experiments/predictions/ensemble/$ENS
# PN_CKPT=$PRJ_DIR/poolnet/results/run-0/models/epoch_9.pth

ENS=ra-oc-p-poc-pnoc-learned-a0.25
CAMS_DIR=experiments/predictions/ensemble/$ENS
PN_CKPT=$PRJ_DIR/poolnet/results/run-1/models/epoch_9.pth

# ENS=ra-oc-p-poc-pnoc-weighted-a0.25
# CAMS_DIR=experiments/predictions/ensemble/$ENS
PN_CKPT=$PRJ_DIR/poolnet/results/run-2/models/epoch_9.pth

CCAMH_TAG=saliency/$DATASET-ccamh-$ARCH@$ENS@b$BATCH_SIZE-fg$FG_T-lr$LR-b$BATCH_SIZE
PN_TAG=$DATASET-pn@ccamh-rs269@$ENS
SAL_PRIORS_DIR=$PRJ_DIR/experiments/predictions/saliency/$CCAMH_TAG@train@scale=0.5,1.0,1.5,2.0@t=0.2@crf=10/

##
## ================================================

ccamh_training
ccamh_inference
ccamh_pseudo_masks_crf

## PoolNet Training and Inference
## ==============================

poolnet_training
poolnet_inference

cp $PN_CKPT $PRJ_DIR/experiments/models/saliency/$PN_TAG.pth
mv $PRJ_DIR/poolnet/results/$PN_TAG $PRJ_DIR/experiments/predictions/saliency/

## Evaluation
## ==============================

CRF_T=10 TAG=$CCAMH_TAG@train@scale=0.5,1.0,1.5,2.0 evaluate_saliency_detection
CRF_T=0 TAG=saliency/$PN_TAG evaluate_saliency_detection

##
## ====================
