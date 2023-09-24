#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=32g
#SBATCH --time=4-24:00:00
#SBATCH --partition=qTRDGPUH
# -p qTRDGPUH,qTRDGPUM
#SBATCH --gres=gpu:V100:1
#SBATCH --account=psy53c17
#SBATCH -J alpha_sweep
#SBATCH -e ./Jobs/reports/report-%A.err
#SBATCH -o ./Jobs/reports/report-%A.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=rio.ohib@gmail.com

sleep 5s

export OMP_NUM_THREADS=1
export MODULEPATH=/apps/Compilers/modules-3.2.10/Debug-Build/Modules/3.2.10/modulefiles/
echo $HOSTNAME >&2 

# source /data/users2/bthapaliya/anaconda-main/anaconda3/bin/activate 
source activate signal

DENSE=0.2
SEED=2022
# for SEED in 110 220 550
for alpha in 0.1 0.2 0.3 0.5 1.0
do
    python main_sailentgrads.py --model 'resnet18' \
    --dataset 'cifar10' \
    --partition_method 'dir' \
    --partition_alpha $alpha \
    --batch_size 16 \
    --lr 0.1 \
    --lr_decay 0.998 \
    --epochs 5 \
    --dense_ratio $DENSE \
    --client_num_in_total 100 --frac 0.1 \
    --comm_round 500 \
    --seed $SEED \
    --exp_name alpha${alpha}_sps0.8
done 

## old seed: 2022
sleep 15s