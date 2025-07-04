#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=48
#SBATCH -p sequana_gpu_shared
#SBATCH -J ss-refine
#SBATCH -o /scratch/lerdl/lucas.david/experiments/logs/ss/refine-%j.out
#SBATCH --time=8:00:00

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
  WORK_DIR=$SCRATCH/wsss-csrm
else
  ENV=local
  WORK_DIR=$HOME/workspace/repos/research/wsss/wsss-csrm
fi

echo "Env:      $ENV"
echo "Work Dir: $WORK_DIR"

# Dataset
DATASET=voc12  # Pascal VOC 2012
# DATASET=coco14  # MS COCO 2014
# DATASET=deepglobe # DeepGlobe Land Cover Classification

. $WORK_DIR/runners/config/env.sh
. $WORK_DIR/runners/config/dataset.sh


## region Segment Anything (Generate binary masks).

cd $WORK_DIR/../segment-anything
export PYTHONPATH=$(pwd)

echo "segment-anything/scripts-amg"
echo "=================================================="

$PIP install -e .

for INDEX in $(seq 0 3)
do
  echo "Enqueuing segment-anything/scripts-amg"
  echo "=================================================="
INDEX=0
CUDA_VISIBLE_DEVICES=$INDEX \
$PY scripts/amg.py \
  --checkpoint ./models/sam_vit_h_4b8939.pth --model-type vit_h \
  --input $DATA_DIR/JPEGImages \
  --output ../SAM_WSSS/SAM/voc12/ &
done
echo "Waiting for jobs to finish..."
wait
echo "All queued segment-anything jobs completed."
## endregion


## region SAM-WSSS (Align priors with SAM masks).

cd ../SAM_WSSS

echo "================================"
echo "SAM_WSSS/main"
echo "================================"

PRED_ROOT=../experiments/predictions

## Pascal VOC 2012:
TAG=csrm/voc12-512-rs101-lr0.007-m0.9-b32-classmix-default-bg0.05-fg0.30-u1-c1-rank3-6-hemfl@rs101p-r1
CLASSES=21

### MS COCO 2014:
# TAG=csrm/coco14-640-rs269-lr0.007-m0.9-b32-colorjitter_classmix-default-bg0.05-fg0.35-u1-c1@rs269pnoc-r1
# CLASSES=81

PSEUDO_PATH=$PRED_ROOT/$TAG@train/pseudos-t0.4-c10
$PY main.py --number_class $CLASSES --pseudo_path $PSEUDO_PATH --sam_path SAM/voc12/
PSEUDO_PATH=$PRED_ROOT/$TAG@val/pseudos-t0.4-c10
$PY main.py --number_class $CLASSES --pseudo_path $PSEUDO_PATH --sam_path SAM/voc12/

## endregion
