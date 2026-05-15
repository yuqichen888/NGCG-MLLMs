#!/bin/sh
#SBATCH --partition=hgnodes
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --cpus-per-task=16
#SBATCH --time=01-23:59:59
#SBATCH --job-name=geo_intern_t2s_h100
#SBATCH --output=%x_%j_geo_intern_t2s_h100.out
#SBATCH --mail-type=ALL


echo "SLURM Job ID from ENV: ${SLURM_JOB_ID}" # For debugging
CUDA_VISIBLE_DEVICES=0 torchrun --standalone --nproc_per_node=1 train.py \
    --config_path config_internvl3_5.yaml \
    --epochs 20 \
    --batch_size 10 \
    --lora True \
    --job_name_from_slurm "geotext_${SLURM_JOB_ID}" \
    --dataset_name 'GeoText' \
    --subset_name 'geo_t2i' \
    --device 'gpu' \
    --warmup_ratio 0.01 \
    --head_num 1 \
    --query 1 \
    --layer 1 \
    --fea_dim 0 \
    --fea_token 'eos' \
    --lr 3e-5
