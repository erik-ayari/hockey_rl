#!/bin/bash
#SBATCH --job-name=TBA_SAC
#SBATCH --cpus-per-task=1
#SBATCH --partition=day
#SBATCH --mem-per-cpu=2G
#SBATCH --gres=gpu:0
#SBATCH --time=00-10:00:00
#SBATCH --error=job.%J.err
#SBATCH --output=job.%J.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=erik.ayari@student.uni-tuebingen.de
singularity run /home/stud402/container.sif python3 -m model.main -c configs/check1_lunar_lander.json
