#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 5
#SBATCH --mem=80g
#SBATCH -p qTRDGPUM
#SBATCH -w arctrdgn002
#SBATCH --gres=gpu:V100:1
#SBATCH -t 03-00
#SBATCH -J sailentgrads
#SBATCH -e error%A.err
#SBATCH -o out%A.out
#SBATCH -A trends53c17
#SBATCH --mail-type=ALL
#SBATCH --mail-user=bthapaliya1@student.gsu.edu
#SBATCH --oversubscribe

sleep 10s

export OMP_NUM_THREADS=1
export MODULEPATH=/apps/Compilers/modules-3.2.10/Debug-Build/Modules/3.2.10/modulefiles/
echo $HOSTNAME >&2 

source /data/users2/bthapaliya/anaconda-main/anaconda3/bin/activate 

python main_fedavg.py --model 'resnet18' \
--dataset 'cifar10' \
--partition_method 'dir' \
--partition_alpha 0.3 \
--batch_size 128 \
--lr 0.1 \
--lr_decay 0.998 \
--epochs 5 \
--client_num_in_total 100 --frac 0.1 \
--comm_round 500 \
--seed 2022

sleep 30s