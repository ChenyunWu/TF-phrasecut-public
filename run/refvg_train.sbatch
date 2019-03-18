#!/bin/bash
#
#SBATCH --job-name=trmi
#SBATCH --output=_rmi/train_refvg_rmi.out  # output file
#SBATCH -e _rmi/train_refvg_rmi.err        # File to which STDERR will be written
#SBATCH --mem=40000
#SBATCH --partition=1080ti-long # Partition to submit to
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1

srun python _rmi/refvg_main.py -m train -w resnet -n RMI -t train -i 750000 -s 50000
