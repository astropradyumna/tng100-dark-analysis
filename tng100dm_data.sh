#!/bin/bash -l
#SBATCH -J RUNewTst
#SBATCH -p saleslab
#SBATCH --ntasks=32
#SBATCH --cpus-per-task=7
#SBATCH --time=48:00:00
#SBATCH --mem=100gb
#SBATCH -o output_log/t100dm1.out
#SBATCH -e output_log/t100dm1.err
#SBATCH --mail-user=psadh003@ucr.edu
#SBATCH --mail-type=ALL

# Load needed modules
# You could also load frequently used modules from within your ~/.bashrc
module load slurm # Should already be loaded
module load openmpi # Should already be loaded
#module load hdf5

# Swtich to the working directory
cd /bigdata/saleslab/psadh003/tng100-dark-analysis

for ((i = 1; i <= 15; i++))
do
    python3 extract_particle_data_within_3rvir.py $i
    python3 tng100dm_data.py $i
done