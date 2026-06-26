#!/usr/bin/bash

#SBATCH -J Segmo_rtrans_Baseline
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=29G
#SBATCH -p batch_grad
#SBATCH -w ariel-v2
#SBATCH -t 2-0
#SBATCH -o /data/dpfla3573/code/Segmo/logs/slurm-%A_rtrans_Baseline.out

cd /data/dpfla3573/code/Segmo
export PYTHONPATH=/data/dpfla3573/code/Segmo:$PYTHONPATH

/data/dpfla3573/anaconda3/envs/momask/bin/python run/train_res_transformer.py \
  --name RTRANS_Baseline \
  --gpu_id 0 \
  --dataset_name t2m \
  --batch_size 64 \
  --seg_captions /data4/local_datasets/HumanML3D/SegmentedCaption \
  --vq_name rvq_nq6_dc512_nc512_noshare_qdp0.2 \
