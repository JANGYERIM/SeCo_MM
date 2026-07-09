#!/usr/bin/bash
#SBATCH -J SeCo_M_P5_V2_4
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=29G
#SBATCH -p batch_grad
#SBATCH -w ariel-v3
#SBATCH -t 4-0
#SBATCH -o /nas2/data/dpfla3573/code/MM_SeCo/logs/slurm-%A_M_P5_V2_4.out

cd /nas2/data/dpfla3573/code/MM_SeCo
export PYTHONPATH=/nas2/data/dpfla3573/code/MM_SeCo:$PYTHONPATH

/nas2/data/dpfla3573/anaconda3/envs/momask/bin/python run/train_t2m_transformer.py \
  --name M_P5_V2-4\
  --gpu_id 0 \
  --dataset_name t2m \
  --batch_size 32 \
  --lambda_feedback 0.2\
  --ipw_alpha 0.2 \
  --vq_name rvq_nq6_dc512_nc512_noshare_qdp0.2 \
