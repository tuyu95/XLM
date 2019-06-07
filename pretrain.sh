#!/usr/bin/env sh
# Distributed under MIT license


#SBATCH -N 1	  # nodes requested
#SBATCH -n 1	  # tasks requested
#SBATCH --partition=LongJobs
#SBATCH --gres=gpu:8
#SBATCH --mem=12000  # memory in Mb
#SBATCH --time=0-80:00:00

export CUDA_HOME=/opt/cuda-9.0.176.1/

export CUDNN_HOME=/opt/cuDNN-7.0/

export STUDENT_ID=$(whoami)

export LD_LIBRARY_PATH=${CUDNN_HOME}/lib64:${CUDA_HOME}/lib64:$LD_LIBRARY_PATH

export LIBRARY_PATH=${CUDNN_HOME}/lib64:$LIBRARY_PATH

export CPATH=${CUDNN_HOME}/include:$CPATH

export PATH=${CUDA_HOME}/bin:${PATH}

export PYTHON_PATH=$PATH


cd /home/s1852803/unmt/XLM

devices=0,1,2,3,4,5,6,7

python train.py \
    --exp_name test_guen_mlm \
    --dump_path /home/s1852803/unmt/XLM/dumped/ \
    --data_path /home/s1852803/unmt/XLM/data/processed/gu-en/ \
    --lgs 'gu-en' \
    --clm_steps '' \
    --mlm_steps 'gu,en' \
    --n_layers 6 \
    --n_heads 8 \
    --dropout 0.1 \
    --attention_dropout 0.1 \
    --gelu_activation true \
    --batch_size 16 \
    --bptt 128 \
    --optimizer adam,lr=0.0001 \
    --epoch_size 200000 \
    --validation_metrics _valid_mlm_ppl \
    --stopping_criterion _valid_mlm_ppl,10


