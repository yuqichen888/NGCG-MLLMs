#!/bin/sh
#SBATCH --partition=nvgpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1  # torchrun handles processes
#SBATCH --cpus-per-task=12    # Request enough CPUs for data loading, DDP, etc.
#SBATCH --gres=gpu:1        # Request 2 GPUs
#SBATCH --mem=32G
#SBATCH --time=01-23:59:59
#SBATCH --job-name=eva
#SBATCH --output=%x_%j_lvec-mini_drone_eva.out
#SBATCH --mail-user=ychen57@uvm.edu
#SBATCH --mail-type=ALL


source ~/scratch/miniconda3/etc/profile.d/conda.sh
conda activate lightvec

module load cuda/12.4.1

cd /gpfs2/scratch/ychen57/code/LightVec
python eva.py \
    --config_path /gpfs2/scratch/ychen57/code/LightVec/config.yaml \
    --checkpoint_to_eval /gpfs2/scratch/ychen57/code/LightVec/src/vlm_backbone/smolvlm/ \
    --eval_batch_size 4 \
    --eval_output_dir ./my_eval_results \
    --device cuda:0

