#!/bin/bash
#SBATCH -A CSC605
#SBATCH -J sae-test
#SBATCH -o %x-%j.out
#SBATCH -t 00:30:00
#SBATCH -p batch
#SBATCH -N 1

module load rocm/6.1.3

cd /lustre/orion/csc605/scratch/george-adams/sae
source activate /lustre/orion/csc605/scratch/george-adams/conda_envs/transformers_env

export MASTER_IP=`ip -f inet addr show hsn0 | sed -En -e 's/.*inet ([0-9.]+).*/\1/p' | head -1`

torchrun --nproc_per_node=8 -m sae --model EleutherAI/pythia-160m --dataset /lustre/orion/csc605/scratch/george-adams/sae/test.hf \
--batch_size 16 --ctx_len 2048 --cache_dir /lustre/orion/csc605/scratch/george-adams/cache_dir_test
