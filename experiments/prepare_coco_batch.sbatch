#!/bin/bash

srun --job-name=rmi_${split} -o rmi_${split}.out -e rmi_${split}.err python build_batches.py -d unc -t ${split}
