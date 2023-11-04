#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=48
#SBATCH -p sequana_gpu_shared
#SBATCH -J ss-train
#SBATCH -o /scratch/lerdl/lucas.david/experiments/logs/ss/train-%j.out
#SBATCH --time=52:00:00

##SBATCH -p sequana_gpu_shared
##SBATCH --ntasks-per-node=48
##SBATCH -p nvidia_long
##SBATCH --ntasks-per-node=24

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

echo "Env:      $ENV"
echo "Work Dir: $WORK_DIR"

# Dataset
DATASET=voc12  # Pascal VOC 2012
# DATASET=coco14  # MS COCO 2014
# DATASET=deepglobe # DeepGlobe Land Cover Classification

. $WORK_DIR/runners/config/env.sh
. $WORK_DIR/runners/config/dataset.sh

cd $WORK_DIR
export PYTHONPATH=$(pwd)

# $PIP install --user -r requirements.txt

## Architecture
### Priors
ARCH=rs269
ARCHITECTURE=resnest269
TRAINABLE_STEM=true
TRAINABLE_BONE=true
DILATED=false
USE_SAL_HEAD=false
MODE=normal
REGULAR=none

LR=0.007  # voc12
# LR=0.004  # coco14
# LR=0.001  # deepglobe

# Training
OPTIMIZER=sgd  # sgd,lion,adam
FIRST_EPOCH=0
EPOCHS=50
BATCH_SIZE=32
ACCUMULATE_STEPS=1

LR_ALPHA_SCRATCH=10.0
LR_ALPHA_BIAS=1.0
LR_POLY_POWER=0.9
GRAD_MAX_NORM=1.

MIXED_PRECISION=true
PERFORM_VALIDATION=true
PROGRESS=true

## Augmentation
AUGMENT=collorjitter
CUTMIX=0
MIXUP=0
LABELSMOOTHING=0

# Evaluation
MIN_TH=0.05
MAX_TH=0.81
CRF_T=0
CRF_GT=0.7


train_ss() {
  echo "=================================================================="
  echo "[train $TAG] started at $(date +'%Y-%m-%d %H:%M:%S')."
  echo "=================================================================="

  WANDB_TAGS="$DATASET,$ARCH,lr:$LR,wd:$WD,ls:$LABELSMOOTHING,b:$BATCH_SIZE,ac:$ACCUMULATE_STEPS,s2c:$S2C_MODE,c2s:$C2S_MODE,warmup:$WARMUP_EPOCHS" \
    WANDB_RUN_GROUP="$DATASET-$ARCH-ss" \
    CUDA_VISIBLE_DEVICES=$DEVICES \
    $PY scripts/train_ss.py \
    --tag $TAG \
    --lr $LR \
    --wd $WD \
    --s2c_mode $S2C_MODE \
    --c2s_mode $C2S_MODE \
    --c2s_sigma $C2S_SIGMA \
    --s2c_sigma $S2C_SIGMA \
    --warmup_epochs $WARMUP_EPOCHS \
    --optimizer $OPTIMIZER \
    --lr_alpha_scratch $LR_ALPHA_SCRATCH \
    --lr_alpha_bias $LR_ALPHA_BIAS \
    --lr_poly_power $LR_POLY_POWER \
    --grad_max_norm $GRAD_MAX_NORM \
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
    --first_epoch $FIRST_EPOCH \
    --max_epoch $EPOCHS \
    --max_steps $MAX_STEPS \
    --dataset $DATASET \
    --data_dir $DATA_DIR \
    --domain_train $DOMAIN_TRAIN \
    --domain_valid $DOMAIN_VALID \
    --progress $PROGRESS \
    --validate $PERFORM_VALIDATION \
    --validate_max_steps $VALIDATE_MAX_STEPS \
    --validate_thresholds $VALIDATE_THRESHOLDS \
    --device $DEVICE \
    --num_workers $WORKERS_TRAIN \
    --restore $RESTORE;
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


LR=0.007
MODE=fix
TRAINABLE_STEM=false
TRAINABLE_BONE=false
ARCHITECTURE=resnest101
ARCH=rs101

EPOCHS=30
MAX_STEPS=74  # 1464 (voc12 train samples) // 16 = 91 steps.
BATCH_SIZE=20
ACCUMULATE_STEPS=1
LABELSMOOTHING=0.1
# AUGMENT=colorjitter # none for DeepGlobe
AUGMENT=cutmix

S2C_MODE=mp
S2C_SIGMA=0.50   # min pixel confidence (conf_p := max_class(prob)_pixel >= S2C_SIGMA)
WARMUP_EPOCHS=1  # min pixel confidence (conf_p := max_class(prob)_pixel >= S2C_SIGMA)
C2S_SIGMA=0.75   # min pixel confidence (conf_p := max_class(prob)_pixel >= S2C_SIGMA)
C2S_MODE=cam

EID=r1  # Experiment ID

# RESTORE=experiments/models/vanilla/voc12-rn50-ra-ls.pth
# RESTORE=experiments/models/vanilla/voc12-rn101-lr0.01-wd0.0001-rals-r1.pth
# RESTORE=experiments/models/vanilla/voc12-rs269-lr0.1-rals-r4.pth
# RESTORE=experiments/models/puzzle/ResNeSt50@Puzzle@optimal.pth
RESTORE=experiments/models/puzzle/ResNeSt101@Puzzle@optimal.pth
# RESTORE=experiments/models/vanilla/voc12-rn101-lr0.1-rals-r1.pth
# RESTORE=experiments/models/pnoc/voc12-rs269-pnoc-b16-lr0.1-ls@rs269-rals-r4.pth
# RESTORE=experiments/models/vanilla/deepglobe-rs50fe-rals-ce-lr0.01-cwnone-r1.pth
# RESTORE=experiments/models/vanilla/deepglobe-rn101-lr0.1-ra-r1.pth
# RESTORE=experiments/models/vanilla/deepglobe-rn101fe-lr0.1-ra-r1.pth

TAG=ss/$DATASET-${ARCH}-lr${LR}-reco-wsss-w_s2c0.1-$AUGMENT-ls-$EID
train_ss

# # DOMAIN=$DOMAIN_TRAIN inference_priors
# DOMAIN=$DOMAIN_VALID inference_priors
# # DOMAIN=$DOMAIN_VALID_SEG inference_priors

# DOMAIN=$DOMAIN_VALID     TAG=$TAG@train@scale=0.5,1.0,1.5,2.0 evaluate_priors
# DOMAIN=$DOMAIN_VALID     TAG=$TAG@train@scale=0.5,1.0,1.5,2.0 CRF_T=10 evaluate_priors
# # DOMAIN=$DOMAIN_VALID_SEG TAG=$TAG@val@scale=0.5,1.0,1.5,2.0   evaluate_priors
