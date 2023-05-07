#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 5
#SBATCH --mem=65g
#SBATCH -p qTRDGPUM
#SBATCH -w arctrddgx001
#SBATCH --gres=gpu:V100:1
#SBATCH -t 03-00
#SBATCH -J c10dispflcifar10
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

python main_dispfl.py --model 'resnet18' \
--dataset 'cifar10' \
--partition_method 'dir' \
--partition_alpha '0.3' \
--batch_size 16 \
--lr 0.1 \
--lr_decay 0.998 \
--epochs 5 \
--client_num_in_total 100 \
--comm_round 500 \
--dense_ratio 0.5 \
--anneal_factor 0.5 \
--seed 2022 \
--cs 'random' \
--dis_gradient_check \
--different_initial

sleep 30s