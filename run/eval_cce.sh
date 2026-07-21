#!/usr/bin/bash

#SBATCH -J CCE_M_P2_V1_4
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=29G
#SBATCH -p batch_grad
#SBATCH -w ariel-v4
#SBATCH -t 1-0
#SBATCH -o /nas2/data/dpfla3573/code/MM_SeCo/logs/slurm-%A_Eval_CCE_PRETRAIN(temp=0.).out

cd /nas2/data/dpfla3573/code/MM_SeCo
export PYTHONPATH=/nas2/data/dpfla3573/code/MM_SeCo:$PYTHONPATH

/nas2/data/dpfla3573/anaconda3/envs/momask/bin/python run/eval_cce.py \
  --name t2m_nlayer8_nhead6_ld384_ff1024_cdp0.1_rvq6ns \
  --gpu_id 0 \
  --dataset_name t2m \
  --which_epoch all \
  --time_steps 10 \
  --temperature 1e-6 \
  --res_name tres_nlayer8_ld384_ff1024_rvq6ns_cdp0.2_sw
