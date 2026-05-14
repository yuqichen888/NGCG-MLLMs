#!/bin/sh
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1  # torchrun handles processes
#SBATCH --cpus-per-task=14    # Request enough CPUs for data loading, DDP, etc.
#SBATCH --mem=128G
#SBATCH --time=00-02:59:59
#SBATCH --job-name=eva
#SBATCH --output=%x_%j_eva-geo-cpu.out
#SBATCH --mail-user=ychen57@uvm.edu
#SBATCH --mail-type=ALL


python eva_compute.py \
   --checkpoint_to_eval  "/gpfs2/scratch/ychen57/code/LightVec/output/checkpoints/smol500_1839233_BS30_LR3e-05_Dgeo_t2i_Aug-geometry_T0.03_ImgRes-512_lr3e-05/smol500_1839233_BS30_LR3e-05_Dgeo_t2i_Aug-geometry_T0.03_ImgRes-512_lr3e-05smol500_1839233_BS30_LR3e-05_Dgeo_t2i_Aug-geometry_T0.03_ImgRes-512_lr3e-05_Epoch-epoch=19.ckpt" \
   --dataset "GeoText" \
   --subset 'geo_all_t2i' \
   --task 'image2text'
