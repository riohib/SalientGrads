#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=64g
#SBATCH --time=4-24:00:00
#SBATCH --partition=qTRDGPU
#SBATCH --gres=gpu:A40:1
#SBATCH --account=psy53c17
#SBATCH -J s80-dispfl
#SBATCH -e ./reports/report-%A.err
#SBATCH -o ./reports/report-%A.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=rio.ohib@gmail.com
# -p qTRDGPUH,qTRDGPUM

sleep 5s

export OMP_NUM_THREADS=1
export MODULEPATH=/apps/Compilers/modules-3.2.10/Debug-Build/Modules/3.2.10/modulefiles/
echo $HOSTNAME >&2 

# source /data/users2/bthapaliya/anaconda-main/anaconda3/bin/activate 
source activate signal

DENSE=0.2
ALPHA=0.3
# SEED=2022
# for alpha in 0.1 0.2 0.3 0.5 1.0
for SEED in 110 220 550
do
    python main_dispfl.py --model 'resnet18' \
    --dataset 'cifar10' \
    --partition_method 'dir' \
    --partition_alpha $ALPHA \
    --batch_size 128 \
    --lr 0.1 \
    --lr_decay 0.998 \
    --epochs 5 \
    --client_num_in_total 100 --frac 0.1 \
    --comm_round 500 \
    --dense_ratio $DENSE \
    --anneal_factor 0.5 \
    --seed 2022 \
    --cs 'random' \
    --dis_gradient_check \
    --different_initial \
    --exp_name alpha${ALPHA}_dense${DENSE}
done

## old seed: 2022
sleep 15s
