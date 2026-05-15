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
   --checkpoint_to_eval "<your path here>" \
   --dataset "GeoText" \
   --subset 'geo_t2i' \
   --task 'image2text'
