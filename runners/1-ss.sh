#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=48
#SBATCH -p sequana_gpu_shared
#SBATCH -J priors
#SBATCH -o /scratch/lerdl/lucas.david/logs/%j-priors.out
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
# Train a model to perform multilabel classification over a WSSS dataset.
#

if [[ "`hostname`" == "sdumont"* ]]; then
  ENV=sdumont
  WORK_DIR=$SCRATCH/single-stage
else
  ENV=local
  WORK_DIR=$HOME/workspace/repos/research/wsss/single-stage
fi

# Dataset
DATASET=voc12  # Pascal VOC 2012
# DATASET=coco14  # MS COCO 2014
# DATASET=deepglobe # DeepGlobe Land Cover Classification

. $WORK_DIR/runners/config/env.sh
. $WORK_DIR/runners/config/dataset.sh

cd $WORK_DIR
export PYTHONPATH=$(pwd)

## Architecture
### Priors
ARCH=rs269
ARCHITECTURE=resnest269
TRAINABLE_STEM=false
DILATED=true
MODE=fix
REGULAR=none
LR=0.007

# Training
# LR=0.1  # defined in dataset.sh
OPTIMIZER=sgd  # sgd,lion
EPOCHS=30
BATCH_SIZE=32
ACCUMULATE_STEPS=1

LR_ALPHA_SCRATCH=10.0
LR_ALPHA_BIAS=1.0

# =========================
# $PIP install lion-pytorch
# OPTIMIZER=lion
# LR_ALPHA_SCRATCH=1.0
# LR_ALPHA_BIAS=1.0
# LR=0.00001
# WD=0.01
# =========================

MIXED_PRECISION=true
PERFORM_VALIDATION=true

## Augmentation
AUGMENT=randaugment  # collorjitter_mixup_cutmix_cutormixup
CUTMIX=0.5
MIXUP=1.0
LABELSMOOTHING=0

## OC-CSE
OC_ARCHITECTURE=$ARCHITECTURE
OC_REGULAR=none
OC_MASK_GN=true # originally done in OC-CSE
OC_TRAINABLE_STEM=true
OC_STRATEGY=random
OC_F_MOMENTUM=0.9
OC_F_GAMMA=2.0
OC_PERSIST=false

## Schedule
P_INIT=0.0
P_ALPHA=4.0
P_SCHEDULE=0.5

OC_INIT=0.3
OC_ALPHA=1.0
OC_SCHEDULE=1.0

OW=1.0
OW_INIT=0.0
OW_SCHEDULE=1.0
OC_TRAIN_MASKS=cams
OC_TRAIN_MASK_T=0.2
OC_TRAIN_INT_STEPS=1

# Evaluation
MIN_TH=0.05
MAX_TH=0.81
CRF_T=0
CRF_GT=0.7


train_singlestage() {
  echo "=================================================================="
  echo "[train $TAG] started at $(date +'%Y-%m-%d %H:%M:%S')."
  echo "=================================================================="

  WANDB_TAGS="$DATASET,$ARCH,lr:$LR,wd:$WD,ls:$LABELSMOOTHING,b:$BATCH_SIZE,ss" \
    WANDB_RUN_GROUP="$DATASET-$ARCH-ss" \
    CUDA_VISIBLE_DEVICES=$DEVICES \
    $PY scripts/train_singlestage.py \
    --tag $TAG \
    --lr $LR \
    --wd $WD \
    --optimizer $OPTIMIZER \
    --lr_alpha_scratch $LR_ALPHA_SCRATCH \
    --lr_alpha_bias $LR_ALPHA_BIAS \
    --batch_size $BATCH_SIZE \
    --accumulate_steps $ACCUMULATE_STEPS \
    --mixed_precision $MIXED_PRECISION \
    --architecture $ARCHITECTURE \
    --dilated $DILATED \
    --mode $MODE \
    --trainable-stem $TRAINABLE_STEM \
    --regularization $REGULAR \
    --restore $RESTORE \
    --image_size $IMAGE_SIZE \
    --min_image_size $MIN_IMAGE_SIZE \
    --max_image_size $MAX_IMAGE_SIZE \
    --augment $AUGMENT \
    --cutmix_prob $CUTMIX \
    --mixup_prob $MIXUP \
    --label_smoothing $LABELSMOOTHING \
    --max_epoch $EPOCHS \
    --dataset $DATASET \
    --data_dir $DATA_DIR \
    --domain_train $DOMAIN_TRAIN \
    --domain_valid $DOMAIN_VALID \
    --validate $PERFORM_VALIDATION \
    --validate_max_steps $VALIDATE_MAX_STEPS \
    --validate_thresholds $VALIDATE_THRESHOLDS \
    --device $DEVICE \
    --num_workers $WORKERS_TRAIN
}

inference_priors() {
  echo "=================================================================="
  echo "[Inference:$TAG] started at $(date +'%Y-%m-%d %H:%M:%S')."
  echo "=================================================================="

  CUDA_VISIBLE_DEVICES=$DEVICES \
    $PY scripts/cam/inference.py \
    --architecture $ARCHITECTURE \
    --regularization $REGULAR \
    --dilated $DILATED \
    --trainable-stem $TRAINABLE_STEM \
    --mode $MODE \
    --tag $TAG \
    --dataset $DATASET \
    --domain $DOMAIN \
    --data_dir $DATA_DIR \
    --device $DEVICE
}

evaluate_priors() {
  WANDB_TAGS="$DATASET,$ARCH,lr:$LR,ls:$LABELSMOOTHING,b:$BATCH_SIZE,ac:$ACCUMULATE_STEPS,domain:$DOMAIN,crf:$CRF_T-$CRF_GT" \
  CUDA_VISIBLE_DEVICES="" \
  $PY scripts/evaluate.py \
    --experiment_name $TAG \
    --dataset $DATASET \
    --domain $DOMAIN \
    --data_dir $DATA_DIR \
    --min_th $MIN_TH \
    --max_th $MAX_TH \
    --crf_t $CRF_T \
    --crf_gt_prob $CRF_GT \
    --mode npy \
    --num_workers $WORKERS_INFER;
}

BATCH_SIZE=16
ACCUMULATE_STEPS=2
LABELSMOOTHING=0.1
AUGMENT=colorjitter  # none for DeepGlobe

EID=r1  # Experiment ID

RESTORE=experiments/models/vanilla/voc12-rs269-lr0.1-rals-r4.pth

TAG=ss/$DATASET-$ARCH-lr$LR-rals-$EID
train_singlestage

# # DOMAIN=$DOMAIN_TRAIN inference_priors
# DOMAIN=$DOMAIN_VALID inference_priors
# # DOMAIN=$DOMAIN_VALID_SEG inference_priors

# DOMAIN=$DOMAIN_VALID     TAG=$TAG@train@scale=0.5,1.0,1.5,2.0 evaluate_priors
# DOMAIN=$DOMAIN_VALID     TAG=$TAG@train@scale=0.5,1.0,1.5,2.0 CRF_T=10 evaluate_priors
# # DOMAIN=$DOMAIN_VALID_SEG TAG=$TAG@val@scale=0.5,1.0,1.5,2.0   evaluate_priors
