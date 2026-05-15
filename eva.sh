#!/bin/sh
#SBATCH --partition=nvgpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1 
#SBATCH --cpus-per-task=12   
#SBATCH --gres=gpu:1       
#SBATCH --mem=32G
#SBATCH --time=01-23:59:59
#SBATCH --job-name=eva
#SBATCH --output=%x_%j_eva.out
#SBATCH --mail-user=ychen57@uvm.edu
#SBATCH --mail-type=ALL


source ~/scratch/miniconda3/etc/profile.d/conda.sh
conda activate ngcg

module load cuda/12.4.1

python eva.py \
    --config_path ./config.yaml \
    --checkpoint_to_eval ./src/vlm_backbone/smolvlm/ \
    --eval_batch_size 4 \
    --eval_output_dir ./my_eval_results \
    --device cuda:0

