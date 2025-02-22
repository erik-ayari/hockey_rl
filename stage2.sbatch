#!/bin/bash
#SBATCH --job-name=TBA_SAC
#SBATCH --cpus-per-task=1
#SBATCH --partition=month
#SBATCH --mem-per-cpu=8G
#SBATCH --gres=gpu:1
#SBATCH --time=13-23:59:59
#SBATCH --error=job.%J.err
#SBATCH --output=job.%J.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=erik.ayari@student.uni-tuebingen.de
singularity run /home/stud402/container.sif python3 -m model.main -c configs/stage2.json --resume_from_checkpoint stage1.ckpt
