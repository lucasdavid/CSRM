#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=12
#SBATCH -p sequana_gpu_shared
#SBATCH -J tr-vanilla
#SBATCH -o /scratch/lerdl/lucas.david/logs/setup-%j.out
#SBATCH --time=01:00:00

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
# Setup.
#

echo "[puzzle/train.sequana] started running at $(date +'%Y-%m-%d %H:%M:%S')."

nodeset -e $SLURM_JOB_NODELIST

cd $SCRATCH/PuzzleCAM

module load sequana/current
# module load gcc/7.4_sequana python/3.9.1_sequana cudnn/8.2_cuda-11.1_sequana
module load gcc/7.4_sequana python/3.8.2_sequana cudnn/8.2_cuda-11.1_sequana

PY=python3.8
PIP=pip3.8

export PYTHONPATH=$(pwd)
# export OMP_NUM_THREADS=16

$PIP install --user torch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu111
$PIP install --user -r requirements.txt
