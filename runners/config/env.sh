# Configurations

## Environment

# if [[ "`hostname`" == "sdumont"* ]]; then
#   ENV=sdumont
# else
#   ENV=local
# fi

if [[ $ENV == sdumont ]]; then
  ### Sdumont
  PY=python3.8
  DEVICE=cuda
  DEVICES=0,1,2,3
  WORKERS_TRAIN=8
  WORKERS_INFER=48
  DATASETS_DIR=$SCRATCH/datasets

  # export OMP_NUM_THREADS=4
  nodeset -e $SLURM_JOB_NODELIST
  module load sequana/current
  module load gcc/7.4_sequana python/3.8.2_sequana cudnn/8.2_cuda-11.1_sequana
else
  ### Local
  PY=python
  DEVICE=cuda
  DEVICES=0
  WORKERS_TRAIN=8
  WORKERS_INFER=8
  DATASETS_DIR=$HOME/workspace/datasets
fi
