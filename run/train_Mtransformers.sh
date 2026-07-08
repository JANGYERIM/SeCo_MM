#!/usr/bin/bash

#SBATCH -J SeCo_M_P6_V1_9
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=29G
#SBATCH -p batch_grad
#SBATCH -w ariel-v1
#SBATCH -t 4-0
#SBATCH -o /nas2/data/dpfla3573/code/MM_SeCo/logs/slurm-%A_M_P6_V1_9.out

cd /nas2/data/dpfla3573/code/MM_SeCo
export PYTHONPATH=/nas2/data/dpfla3573/code/MM_SeCo:$PYTHONPATH

/nas2/data/dpfla3573/anaconda3/envs/momask/bin/python run/train_t2m_transformer.py \
  --name M_P6_V1_9\
  --gpu_id 0 \
  --dataset_name t2m \
  --batch_size 32 \
  --consistency_type motion_lookup \
  --lambda_consistency 0.5 \
  --vq_name rvq_nq6_dc512_nc512_noshare_qdp0.2 \
  --debug_print_topk \
  --debug_topk 5 \
  --debug_num_motions 3\
  --debug_num_positions 3\
 # --lambda_margin 0.3 \