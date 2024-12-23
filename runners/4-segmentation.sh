#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=48
#SBATCH -p sequana_gpu_shared
#SBATCH -J segm
#SBATCH -o /scratch/lerdl/lucas.david/logs/%j-segm.out
#SBATCH --time=32:00:00

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

if [[ "$(hostname)" == "sdumont"* ]]; then
  ENV=sdumont
  WORK_DIR=$SCRATCH/pnoc
else
  ENV=local
  WORK_DIR=$HOME/workspace/repos/research/wsss/pnoc
fi

# Dataset
DATASET=voc12 # Pascal VOC 2012
# DATASET=coco14  # MS COCO 2014
# DATASET=deepglobe # DeepGlobe Land Cover Classification

. $WORK_DIR/runners/config/env.sh
. $WORK_DIR/runners/config/dataset.sh

cd $WORK_DIR
export PYTHONPATH=$(pwd)


## Architecture
ARCHITECTURE=deeplabv3p
ARCH=rs269
BACKBONE=resnest269
PRETRAINED_WEIGHTS=imagenet

# ARCHITECTURE=segformer
# BACKBONE=mit_b0
# PRETRAINED_WEIGHTS=./experiments/models/pretrained/mit_b0.pth
# ARCH=segformer_mit_b0
# BACKBONE=mit_b2
# PRETRAINED_WEIGHTS=./experiments/models/pretrained/mit_b2.pth
# ARCH=segformer_mit_b2
# BACKBONE=mit_b5
# PRETRAINED_WEIGHTS=./experiments/models/pretrained/mit_b5.pth
# ARCH=segformer_mit_b5

# BACKBONE=swin_l
# ARCH=swin_l_22k
# PRETRAINED_WEIGHTS=./experiments/models/pretrained/swin_large_patch4_window7_224_22k.pth


GROUP_NORM=true
DILATED=false
MODE=normal
TRAIN_STEM=true
TRAIN_BONE=true

OPTIMIZER=momentum
# LR=0.01 # SegFormer
# WD=0.01
WD=4e-05
LR=0.007 # voc12
# LR=0.004  # coco14
# LR=0.01  # deepglobe

EPOCHS=50

BATCH_SIZE=32
ACCUMULATE_STEPS=1

AUGMENT=none # colorjitter_randaug_cutmix_mixup_cutormixup
AUG=no
CUTMIX=0.5
MIXUP=1.
LABELSMOOTHING=0 # 0.1

# Infrastructure
MIXED_PRECISION=true

CRF_T=10
CRF_GT=1
MIN_TH=0.20
MAX_TH=0.81

# RESTORE=/path/to/weights
# RESTORE_STRICT=true

segm_training() {
  echo "================================================="
  echo "Semantic Segmentation Training $TAG"
  echo "================================================="

  WANDB_TAGS="$DATASET,$ARCH,segmentation,b:$BATCH_SIZE,gn,lr:$LR,opt:$OPTIMIZER,ls:$LABELSMOOTHING,aug:$AUG,mode:$MODE" \
    WANDB_RUN_GROUP="$DATASET-$ARCH-segmentation" \
    CUDA_VISIBLE_DEVICES=$DEVICES \
    $PY scripts/segmentation/train.py \
    --tag $TAG \
    --optimizer $OPTIMIZER \
    --lr $LR \
    --max_epoch $EPOCHS \
    --batch_size $BATCH_SIZE \
    --accumulate_steps $ACCUMULATE_STEPS \
    --mixed_precision $MIXED_PRECISION \
    --architecture $ARCHITECTURE \
    --backbone $BACKBONE \
    --trainable-stem $TRAIN_STEM \
    --trainable-backbone $TRAIN_BONE \
    --dilated $DILATED \
    --mode $MODE \
    --backbone_weights $PRETRAINED_WEIGHTS \
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
    --architecture $ARCHITECTURE \
    --backbone $BACKBONE \
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
  # dCRF is not re-applied during evaluation (crf_t=0)

  CUDA_VISIBLE_DEVICES="" \
    WANDB_TAGS="$DATASET,domain:$DOMAIN_VALID,$ARCH,segmentation,b:$BATCH_SIZE,gn,lr:$LR,opt:$OPTIMIZER,ls:$LABELSMOOTHING,aug:$AUGMENT,mode:$MODE,crf:$CRF_T-$CRF_GT" \
    WANDB_RUN_GROUP="$DATASET-$ARCH-segmentation" \
    $PY scripts/evaluate.py \
    --experiment_name $TAG \
    --pred_dir $SEGM_PRED_DIR \
    --dataset $DATASET \
    --domain $DOMAIN \
    --data_dir $DATA_DIR \
    --min_th $MIN_TH \
    --max_th $MAX_TH \
    --crf_gt_prob $CRF_GT \
    --num_workers $WORKERS_INFER \
    --crf_t 0 \
    --mode png
}

## 4.1 DeepLabV3+ Training
##

LABELSMOOTHING=0.1
AUGMENT=none  # colorjitter_randaug_cutmix_mixup_cutormixup
AUG=no

## For supervised segmentation:
# PRIORS_TAG=sup
# MASKS_DIR=""

## For custom masks (pseudo masks from WSSS):

## Pascal VOC 2012:
PRIORS_TAG=csrm-rs101-hemfl-sam
MASKS_DIR=./experiments/predictions/csrm/voc12-512-rs101-lr0.007-m0.9-b32-classmix-default-bg0.05-fg0.30-u1-c1-rank3-6-hemfl@rs101p-r1_pseudos-t0.4-c10__max_iou_imp2

## MS COCO 2014:
# PRIORS_TAG=csrm-rs269-ccamh-sam
# MASKS_DIR=./experiments/predictions/csrm/coco14-640-rs269-lr0.007-m0.9-b32-colorjitter_classmix-default-bg0.05-fg0.35-u1-c1@rs269pnoc-r1_pseudos-t0.4-c10__max_iou_imp2

TAG=segmentation/$DATASET-$IMAGE_SIZE-$ARCH-lr$LR-b$BATCH_SIZE-$MODE-$AUG-$PRIORS_TAG
segm_training

# 4.2 DeepLabV3+ Inference
#

SEGM_PRED_DIR=./experiments/predictions/$TAG@crf=$CRF_T
#  DOMAIN=$DOMAIN_VALID     segm_inference
DOMAIN=$DOMAIN_VALID_SEG segm_inference
DOMAIN=$DOMAIN_TEST      SEGM_PRED_DIR=./experiments/predictions/$TAG@test@crf=$CRF_T segm_inference

# 4.3. Evaluation
#
# DOMAIN=$DOMAIN_VALID     evaluate_masks
DOMAIN=$DOMAIN_VALID_SEG evaluate_masks
