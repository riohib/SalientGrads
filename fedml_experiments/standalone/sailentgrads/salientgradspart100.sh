#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 5
#SBATCH --mem=44g
#SBATCH --partition=qTRDGPUH
#SBATCH --gres=gpu:V100:1
#SBATCH -t 01-06
#SBATCH -e error-1%A.err
#SBATCH -o out-1%A.out
#SBATCH --account=psy53c17
#SBATCH -J rohib
#SBATCH --mail-type=ALL
#SBATCH --mail-user=bthapaliya1@student.gsu.edu
#SBATCH --oversubscribe

sleep 10s

export OMP_NUM_THREADS=1
export MODULEPATH=/apps/Compilers/modules-3.2.10/Debug-Build/Modules/3.2.10/modulefiles/
echo $HOSTNAME >&2 

source /data/users2/bthapaliya/anaconda-main/anaconda3/bin/activate 

python main_sailentgrads.py --model 'resnet18' \
--dataset 'cifar100' \
--partition_method 'my_part' \
--partition_alpha 0.2 \
--batch_size 16 \
--lr 0.1 \
--lr_decay 0.998 \
--epochs 5 \
--dense_ratio 0.5 \
--client_num_in_total 100 --frac 0.1 \
--comm_round 500 \
--seed 2022

sleep 30s
