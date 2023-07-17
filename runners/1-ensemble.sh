#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=48
#SBATCH -p sequana_gpu_shared
#SBATCH -J priors
#SBATCH -o /scratch/lerdl/lucas.david/logs/%j-priors.out
#SBATCH --time=96:00:00

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

if [[ "$(hostname)" == "sdumont"* ]]; then
  ENV=sdumont
  WORK_DIR=$SCRATCH/wsss-ensemble
else
  ENV=local
  WORK_DIR=/home/ldavid/workspace/repos/research/wsss-ensemble
fi

# Dataset
DATASET=voc12 # Pascal VOC 2012
# DATASET=coco14  # MS COCO 2014
# DATASET=deepglobe # DeepGlobe Land Cover Classification

. $WORK_DIR/runners/config/env.sh
. $WORK_DIR/runners/config/dataset.sh
. $WORK_DIR/runners/config/wandb.sh

cd $WORK_DIR
export PYTHONPATH=$(pwd)

ALPHA=0.25
MERGE=avg
WEIGHTS_PATH=""


run_ensemble() {
  CUDA_VISIBLE_DEVICES=$DEVICES \
    $PY scripts/ensemble.py \
    --tag $TAG \
    --merge $MERGE \
    --alpha $ALPHA \
    --weights_path "$WEIGHTS_PATH" \
    --dataset $DATASET \
    --domain $DOMAIN \
    --data_dir $DATA_DIR \
    --num_workers $WORKERS_INFER \
    --experiments $EXPERIMENTS
}


# TAG=ensemble/ra-oc-p-avg
# EXPERIMENTS="vanilla/resnest269@randaug@train@scale=0.5,1.0,1.5,2.0 literature/rn38d-occse resnest269@puzzlerep2@train@scale=0.5,1.0,1.5,2.0"
# MERGE=avg
# WEIGHTS_PATH=""
# run_ensemble

# TAG=ensemble/ra-oc-l2g-p-pnoc-avg
# EXPERIMENTS="vanilla/resnest269@randaug@train@scale=0.5,1.0,1.5,2.0 literature/rn38d-occse literature/l2g resnest269@puzzlerep2@train@scale=0.5,1.0,1.5,2.0 pnoc/voc12-rs269-pnoc-b16-lr0.1-ls@rs269-lsra-r4@train@scale=0.5,1.0,1.5,2.0"
# MERGE=avg
# run_ensemble

TAG=ensemble/ra-oc-p-poc-pnoc-avg
EXPERIMENTS="vanilla/resnest269@randaug@train@scale=0.5,1.0,1.5,2.0 literature/rn38d-occse resnest269@puzzlerep2@train@scale=0.5,1.0,1.5,2.0 poc/voc12-rs269-poc-ls0.1@rs269ra-r3@train@scale=0.5,1.0,1.5,2.0 pnoc/voc12-rs269-pnoc-b16-lr0.1-ls@rs269-lsra-r4@train@scale=0.5,1.0,1.5,2.0"
MERGE=avg

# DOMAIN=$DOMAIN_TRAIN run_ensemble

DOMAIN=$DOMAIN_VALID run_ensemble

# TAG=ensemble/ra-oc-p-poc-pnoc-highest
# EXPERIMENTS="vanilla/resnest269@randaug@train@scale=0.5,1.0,1.5,2.0 literature/rn38d-occse resnest269@puzzlerep2@train@scale=0.5,1.0,1.5,2.0 poc/voc12-rs269-poc-ls0.1@rs269ra-r3@train@scale=0.5,1.0,1.5,2.0 pnoc/voc12-rs269-pnoc-b16-lr0.1-ls@rs269-lsra-r4@train@scale=0.5,1.0,1.5,2.0"
# MERGE=highest
# WEIGHTS_PATH=experiments/logs/ious.csv
# run_ensemble

# TAG=ensemble/ra-oc-p-poc-pnoc-ranked
# MERGE=ranked
# ALPHA=2.0
# run_ensemble

# TAG=ensemble/ra-oc-p-poc-pnoc-weighted
# MERGE=weighted
# ALPHA=0.25
# run_ensemble

# MERGE=learned
# ALPHA=0.25
# TAG=ensemble/ra-oc-p-poc-pnoc-$MERGE-a$ALPHA
# run_ensemble

# WEIGHTS=puzzle/ResNeSt269@Puzzle@optimal
# WEIGHTS=puzzle/resnest269@puzzlerep2
# WEIGHTS=puzzle/resnest269@puzzlerep22
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
