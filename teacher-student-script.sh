#!bin/bash

#SBATCH --job-name="Teacher student base"
#SBATCH --time=08:00:00
#SBATCH --ntasks=8
#SBATCH --partition=gpu-a100
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=1G
#SBATCH --account=education-eemcs-courses-cse3000

module load 2023r1
module load cuda/11.6
module load python

python -m pip install -r requirements.txt
python convert.py

srun python teacher-student.py > ts.log
