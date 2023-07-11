#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=48
#SBATCH -p sequana_gpu_shared
#SBATCH -J dg-segm
#SBATCH -o /scratch/lerdl/lucas.david/logs/%j-dg-segm.out
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
# Segmentation with Pseudo Semantic Segmentation Masks
#

# ENV=sdumont
# WORK_DIR=$SCRATCH/PuzzleCAM
ENV=local
WORK_DIR=$HOME/workspace/repos/research/pnoc

# Dataset
# DATASET=voc12  # Pascal VOC 2012
# DATASET=coco14  # MS COCO 2014
DATASET=deepglobe # DeepGlobe Land Cover Classification

. $WORK_DIR/runners/config/env.sh
. $WORK_DIR/runners/config/dataset.sh

cd $WORK_DIR
export PYTHONPATH=$(pwd)

# Architecture
# ARCH=rs269
# ARCHITECTURE=resnest269
ARCH=rs101
ARCHITECTURE=resnest101
GROUP_NORM=true
DILATED=false
MODE=fix

# LR=0.007  # voc12
# LR=0.004  # coco14
# LR=0.001  # deepglobe

IMAGE_SIZE=320
MIN_IMAGE_SIZE=$IMAGE_SIZE
MAX_IMAGE_SIZE=$IMAGE_SIZE

EPOCHS=50

BATCH_SIZE=16
ACCUMULATE_STEPS=2

AUGMENT=none # colorjitter_randaug_cutmix_mixup_cutormixup
CUTMIX=0.5
MIXUP=1.
LABELSMOOTHING=0 # 0.1

# Infrastructure
MIXED_PRECISION=true

CRF_T=0
CRF_GT=0.9
MIN_TH=0.20
MAX_TH=0.81

# RESTORE=/path/to/weights
# RESTORE_STRICT=true

segm_training() {
  echo "================================================="
  echo "Semantic Segmentation Training $TAG"
  echo "================================================="

  WANDB_TAGS="$DATASET,$ARCH,segmentation,b:$BATCH_SIZE,gn,lr:$LR,ls:$LABELSMOOTHING,aug:$AUGMENT" \
    WANDB_RUN_GROUP="$DATASET-$ARCH-segmentation" \
    CUDA_VISIBLE_DEVICES=$DEVICES \
    $PY scripts/segmentation/train.py \
    --tag $TAG \
    --lr $LR \
    --max_epoch $EPOCHS \
    --batch_size $BATCH_SIZE \
    --accumulate_steps $ACCUMULATE_STEPS \
    --mixed_precision $MIXED_PRECISION \
    --architecture $ARCHITECTURE \
    --dilated $DILATED \
    --mode $MODE \
    --use_gn $GROUP_NORM \
    --image_size $IMAGE_SIZE \
    --min_image_size $MIN_IMAGE_SIZE \
    --max_image_size $MAX_IMAGE_SIZE \
    --augment "$AUGMENT" \
    --cutmix_prob $CUTMIX \
    --mixup_prob $MIXUP \
    --label_smoothing $LABELSMOOTHING \
    --dataset $DATASET \
    --data_dir $DATA_DIR \
    --masks_dir "$MASKS_DIR" \
    --domain_train $DOMAIN_TRAIN \
    --domain_valid $DOMAIN_VALID_SEG \
    --num_workers $WORKERS_TRAIN
}

segm_inference() {
  echo "================================================="
  echo "Semantic Segmentation Inference $TAG"
  echo "================================================="

  CUDA_VISIBLE_DEVICES=$DEVICES \
    $PY scripts/segmentation/inference.py \
    --tag $TAG \
    --pred_dir $SEGM_PRED_DIR \
    --backbone $ARCHITECTURE \
    --mode $MODE \
    --dilated $DILATED \
    --use_gn $GROUP_NORM \
    --dataset $DATASET \
    --domain $DOMAIN \
    --crf_t $CRF_T \
    --crf_gt_prob $CRF_GT \
    --data_dir $DATA_DIR \
    --num_workers $WORKERS_TRAIN
}

evaluate_masks() {
  CUDA_VISIBLE_DEVICES="" \
    WANDB_RUN_GROUP="$W_GROUP" \
    WANDB_TAGS="$DATASET,domain:$DOMAIN,$ARCH,ccamh,rw,segmentation,crf:$CRF_T-$CRF_GT" \
    $PY scripts/evaluate.py \
    --experiment_name $TAG \
    --dataset $DATASET \
    --domain $DOMAIN \
    --data_dir $DATA_DIR \
    --min_th $MIN_TH \
    --max_th $MAX_TH \
    --crf_t $CRF_T \
    --crf_gt_prob $CRF_GT \
    --mode $EVAL_MODE \
    --num_workers $WORKERS_INFER
}

## 4.1 DeepLabV3+ Training
##

LABELSMOOTHING=0.1

## For supervised segmentation:
PRIORS_TAG=sup
MASKS_DIR=""

## For custom masks (pseudo masks from WSSS):
# PRIORS_TAG=pnoc-ccamh-rw
# MASKS_DIR=./experiments/predictions/rw/$DATASET-an@ccamh@rs269-pnoc@beta=10@exp_times=8@rw@crf=$CRF_T

TAG=segmentation/$DATASET-$IMAGE_SIZE-d3p-lr$LR-ls-$PRIORS_TAG
# segm_training

## 4.2 DeepLabV3+ Inference
##

# CRF_T=1 DOMAIN=$DOMAIN_TRAIN SEGM_PRED_DIR=./experiments/predictions/$TAG segm_inference
# CRF_T=1 DOMAIN=$DOMAIN_VALID_SEG SEGM_PRED_DIR=./experiments/predictions/$TAG segm_inference
# CRF_T=1 DOMAIN=$DOMAIN_TEST SEGM_PRED_DIR=./experiments/predictions/$TAG segm_inference

## 4.3. Evaluation
##

# DOMAIN=$DOMAIN_VALID_SEG EVAL_MODE=png evaluate_masks

# region Deep-Globe Land Cover
## LR and augmentation search
## ==========================
LRS="0.1 0.01 0.001"
AUGMENTS="none clahe clahe_cutmix"

for LR in $LRS; do
  for AUGMENT in $AUGMENTS; do
    TAG=segmentation/$DATASET-$IMAGE_SIZE-d3p-lr$LR-ls-$AUGMENT-sup
    # segm_training
  done
done

# endregion
