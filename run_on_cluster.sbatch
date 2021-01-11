#!/bin/bash
#SBATCH -J Magrathea
#SBATCH -o Magrathea_%j.txt
#SBATCH -e Magrathea_%j.err
#SBATCH -p skx-dev
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=48
#SBATCH --export=ALL
#SBATCH --time=00:30:00
#SBATCH -A TG-EAR170019
#SBATCH --mail-user=hrmd@mit.edu
#SBATCH --mail-type=all
module add matlab
python3 run_on_cluster.py 
