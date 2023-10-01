#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=48
#SBATCH -p sequana_gpu_shared
#SBATCH -J ss-train
#SBATCH -o /scratch/lerdl/lucas.david/logs/%j-ss.out
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
TRAINABLE_STEM=true
TRAINABLE_BONE=true
DILATED=false
USE_SAL_HEAD=true
MODE=normal
REGULAR=none

LR=0.007  # voc12
# LR=0.004  # coco14
# LR=0.001  # deepglobe

# Training
OPTIMIZER=sgd  # sgd,lion
EPOCHS=50
BATCH_SIZE=32
ACCUMULATE_STEPS=1

LR_ALPHA_SCRATCH=10.0
LR_ALPHA_BIAS=1.0
LR_POLY_POWER=0.9

MIXED_PRECISION=true
PERFORM_VALIDATION=true
PROGRESS=true

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


train_ss() {
  echo "=================================================================="
  echo "[train $TAG] started at $(date +'%Y-%m-%d %H:%M:%S')."
  echo "=================================================================="

  WANDB_TAGS="$DATASET,$ARCH,lr:$LR,wd:$WD,ls:$LABELSMOOTHING,b:$BATCH_SIZE,ac:$ACCUMULATE_STEPS,m:$S2C_MODE,l:$C2S_PSEUDO_LABEL_MODE" \
    WANDB_RUN_GROUP="$DATASET-$ARCH-ss" \
    CUDA_VISIBLE_DEVICES=$DEVICES \
    $PY scripts/train_ss.py \
    --tag $TAG \
    --lr $LR \
    --wd $WD \
    --s2c_mode $S2C_MODE \
    --c2s_pseudo_label_mode $C2S_PSEUDO_LABEL_MODE \
    --c2s_sigma $C2S_SIGMA \
    --s2c_sigma $S2C_SIGMA \
    --optimizer $OPTIMIZER \
    --lr_alpha_scratch $LR_ALPHA_SCRATCH \
    --lr_alpha_bias $LR_ALPHA_BIAS \
    --lr_poly_power $LR_POLY_POWER \
    --batch_size $BATCH_SIZE \
    --accumulate_steps $ACCUMULATE_STEPS \
    --mixed_precision $MIXED_PRECISION \
    --architecture $ARCHITECTURE \
    --use_sal_head $USE_SAL_HEAD \
    --dilated $DILATED \
    --mode $MODE \
    --trainable-stem $TRAINABLE_STEM \
    --trainable-backbone $TRAINABLE_BONE \
    --regularization $REGULAR \
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
    --domain_train $DOMAIN_VALID \
    --domain_valid $DOMAIN_VALID_SEG \
    --progress $PROGRESS \
    --validate $PERFORM_VALIDATION \
    --validate_max_steps $VALIDATE_MAX_STEPS \
    --validate_thresholds $VALIDATE_THRESHOLDS \
    --device $DEVICE \
    --restore $RESTORE \
    --num_workers $WORKERS_TRAIN;

  # TODO: fix these domains later.

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

LR=0.007  # voc12
MODE=fix
TRAINABLE_STEM=false
TRAINABLE_BONE=false
ARCHITECTURE=resnest101
ARCH=rs101
OC_ARCHITECTURE=$ARCHITECTURE

EPOCHS=10
BATCH_SIZE=16
ACCUMULATE_STEPS=2
LABELSMOOTHING=0
AUGMENT=colorjitter  # none for DeepGlobe

S2C_MODE=bce
C2S_PSEUDO_LABEL_MODE=cam
S2C_SIGMA=1.0
C2S_SIGMA=0.5

EID=r1  # Experiment ID

# RESTORE=experiments/models/vanilla/voc12-rn50-ra-ls.pth
# RESTORE=experiments/models/vanilla/voc12-rn101-lr0.01-wd0.0001-rals-r1.pth
# RESTORE=experiments/models/vanilla/voc12-rs269-lr0.1-rals-r4.pth
# RESTORE=experiments/models/puzzle/ResNeSt50@Puzzle@optimal.pth
RESTORE=experiments/models/puzzle/ResNeSt101@Puzzle@optimal.pth
# RESTORE=experiments/models/pnoc/voc12-rs269-pnoc-b16-lr0.1-ls@rs269-lsra-r4.pth
# RESTORE=experiments/models/vanilla/deepglobe-rs50fe-rals-ce-lr0.01-cwnone-r1.pth
# RESTORE=experiments/models/vanilla/deepglobe-rn101-lr0.1-ra-r1.pth
# RESTORE=experiments/models/vanilla/deepglobe-rn101fe-lr0.1-ra-r1.pth

# # KL Divergence(concat(SEG[bg]), CAMs), SEG) (mIoU=NaN)
# S2C_MODE="kld"
# S2C_SIGMA=10.0  # KL temperature
# USE_SAL_HEAD=false
# TAG=ss/$DATASET-$ARCH-lr$LR-seggt_wu0-$S2C_MODE-$EID
# train_ss

# # KL Divergence(concat(SEG[bg]), CAMs), SEG) (mIoU=NaN)
# S2C_MODE="kld"
# S2C_SIGMA=10.0  # KL temperature
# USE_SAL_HEAD=true
# TAG=ss/$DATASET-$ARCH-lr$LR-seggt_wu0-$S2C_MODE-$EID
# train_ss

# # BCE(CAMs, 1 - SEG[bg]) (mIoU=70.414)
# S2C_MODE="bce"
# S2C_SIGMA=0.9  # initial min bg pixel confidence (conf_p := prob[bg] >= lerp(S2C_SIGMA, 0.5, 1))
# USE_SAL_HEAD=false
# TAG=ss/$DATASET-$ARCH-lr$LR-seggt_wu0-$S2C_MODE-$EID
# train_ss

# # Branches Mutual Promotion >> sum(CE(concat(SEG[bg]), CAMs), SEG) * conf_p) / conf_p.count (mIoU=70.414)
# S2C_MODE="mp"
# S2C_SIGMA=0.5  # min pixel confidence (conf_p := max_class(prob)_pixel >= S2C_SIGMA)
# USE_SAL_HEAD=false
# TAG=ss/$DATASET-$ARCH-lr$LR-sal-seggt_wu0-$S2C_MODE-$EID
# train_ss

# # Branches Mutual Promotion >> sum(CE(concat(SEG[bg]), CAMs), SEG) * conf_p) / conf_p.count (mIoU=70.791)
# S2C_MODE="mp"
# S2C_SIGMA=0.5  # min pixel confidence (conf_p := max_class(prob)_pixel >= S2C_SIGMA)
# USE_SAL_HEAD=true
# TAG=ss/$DATASET-$ARCH-lr$LR-sal-seggt_wu0-$S2C_MODE-$EID
# train_ss

# Branches Mutual Promotion >> sum(CE(concat(SEG[bg]), CAMs), SEG) * conf_p) / conf_p.count (mIoU=?)
S2C_MODE="mp"
S2C_SIGMA=0.5  # min pixel confidence (conf_p := max_class(prob)_pixel >= S2C_SIGMA)
USE_SAL_HEAD=false
LR_POLY_POWER=0.9
TAG=ss/$DATASET-$ARCH-lr${LR}c-pw$LR_POLY_POWER-wu0-$S2C_MODE-$EID
# train_ss

# Branches Mutual Promotion >> sum(CE(concat(SEG[bg]), CAMs), SEG) * conf_p) / conf_p.count (mIoU=?)
S2C_MODE="mp"
S2C_SIGMA=0.5  # min pixel confidence (conf_p := max_class(prob)_pixel >= S2C_SIGMA)
USE_SAL_HEAD=true
LR_POLY_POWER=0.0
TAG=ss/$DATASET-$ARCH-lr${LR}c-wu0-$S2C_MODE-$EID
# train_ss


S2C_MODE="mp"
S2C_SIGMA=0.5  # min pixel confidence (conf_p := max_class(prob)_pixel >= S2C_SIGMA)
USE_SAL_HEAD=false
LR_POLY_POWER=0.9
DILATED=true
TAG=ss/$DATASET-${ARCH}d-lr${LR}-wu0-$S2C_MODE-$EID
# train_ss

C2S_PSEUDO_LABEL_MODE=mp
S2C_SIGMA=0.5  # min pixel confidence (conf_p := max_class(prob)_pixel >= S2C_SIGMA)
C2S_SIGMA=0.75  # min pixel confidence (conf_p := max_class(prob)_pixel >= S2C_SIGMA)
TAG=ss/$DATASET-${ARCH}d-lr${LR}-lm_$C2S_PSEUDO_LABEL_MODE-wu0-$S2C_MODE-$EID
train_ss

# # DOMAIN=$DOMAIN_TRAIN inference_priors
# DOMAIN=$DOMAIN_VALID inference_priors
# # DOMAIN=$DOMAIN_VALID_SEG inference_priors

# DOMAIN=$DOMAIN_VALID     TAG=$TAG@train@scale=0.5,1.0,1.5,2.0 evaluate_priors
# DOMAIN=$DOMAIN_VALID     TAG=$TAG@train@scale=0.5,1.0,1.5,2.0 CRF_T=10 evaluate_priors
# # DOMAIN=$DOMAIN_VALID_SEG TAG=$TAG@val@scale=0.5,1.0,1.5,2.0   evaluate_priors
