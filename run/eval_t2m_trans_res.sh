#!/usr/bin/bash

#SBATCH -J SeCo_Eval_M_P2_V1_2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=29G
#SBATCH -p batch_grad
#SBATCH -w ariel-v2
#SBATCH -t 1-0
#SBATCH -o /nas2/data/dpfla3573/code/MM_SeCo/logs/slurm-%A_Eval_M_P2_V1_6.out

cd /nas2/data/dpfla3573/code/MM_SeCo
export PYTHONPATH=/nas2/data/dpfla3573/code/MM_SeCo:$PYTHONPATH

/nas2/data/dpfla3573/anaconda3/envs/momask/bin/python run/eval_t2m_trans_res.py \
  --name M_P2_V1_6 \
  --gpu_id 0 \
  --use_res_model \
  --dataset_name t2m \
  --which_epoch all \
  --time_steps 10 \
  --res_name tres_nlayer8_ld384_ff1024_rvq6ns_cdp0.2_sw
