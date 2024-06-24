#!/bin/bash

#SBATCH --job-name="Teacher student base"
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --partition=gpu-a100
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-cpu=2G
#SBATCH --account=education-eemcs-courses-cse3000

module load 2023r1
module load openmpi
module load cuda/12.1
module load python

# python -m venv ./venv
# /scratch/zliutkus/research-project/venv/bin/pip install -r requirements.txt
# /scratch/zliutkus/research-project/venv/bin/python convert.py

srun /scratch/zliutkus/research-project/venv/bin/python teacher-student.py > ts-50.log

