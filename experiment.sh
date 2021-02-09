#!/bin/bash

#SBATCH -J  SAM         # Job name
#SBATCH -o  %j.out    # Name of stdout output file (%j expands to %jobId)
#SBATCH -t  10:00:00            # Run time (hh:mm:ss) - 1.5 hours

#### Select  GPU
# SBATCH -p 2080ti          # queue  name  or  partiton name gpu-titanxp, gpu-2080ti

## gpu 2장
##SBATCH   --gres=gpu:6

## 노드 지정하지않기
#SBATCH   --nodes=1

## gpu 가 2장이면  --ntasks=2, --tasks-per-node=2 , --cpus-per-task=1
## gpu 가 4장이면  --ntasks=4, --tasks-per-node=4 , --cpus-per-task=1

#SBTACH   --ntasks=6
#SBATCH   --tasks-per-node=6
#SBATCH   --cpus-per-task=1

# WORLD_SIZE_JOB=\$SLURM_NTASKS
# RANK_NODE=\$SLURM_NODEID
# PROC_PER_NODE=

# DDP_BACKEND=c10d


export MASTER_ADDR=$(hostname)
export MASTER_PORT="6819"
cd  $SiLURM_SUBMIT_DIR

echo "SLURM_SUBMIT_DIR=$SLURM_SUBMIT_DIR"
echo "CUDA_HOME=$CUDA_HOME"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "CUDA_VERSION=$CUDA_VERSION"


srun -l /bin/hostname
srun -l /bin/pwd
srun -l /bin/date

module  purge
module  load   postech

echo "source  $HOME/anaconda3/etc/profile.d/conda.sh"
source $HOME/anaconda3/etc/profile.d/conda.sh

conda activate pytorch
python /home/junyoon/project/cvlab-project/trainer.py 

date

squeue  --job  $SLURM_JOBID

echo  "##### END #####"
