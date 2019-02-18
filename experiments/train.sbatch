#!/bin/bash
#
#SBATCH --job-name=train
#SBATCH --output=train_unc.out  # output file
#SBATCH -e train_unc.err        # File to which STDERR will be written
#SBATCH --partition=titanx-long # Partition to submit to 
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1

srun python main.py -m train -w resnet -n RMI -d unc -t train -g 0 -i 750000 -s 50000
