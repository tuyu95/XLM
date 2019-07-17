#!/bin/sh

#SBATCH -N 1	  # nodes requested
#SBATCH -n 1	  # tasks requested
#SBATCH --partition=General_Usage
#SBATCH --gres=gpu:2
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

source /home/${STUDENT_ID}/miniconda3/bin/activate mlp

devices=0,1

CUDA_VISIBLE_DEVICES=$devices python3 /home/s1852803/unmt/XLM/train.py \
    --exp_name supMT_engu \
    --dump_path /home/s1852803/unmt/XLM/dumped/ \
    --reload_model /home/s1852803/unmt/XLM/best-valid_mlm_ppl.pth,/home/s1852803/unmt/XLM/best-valid_mlm_ppl.pth \
    --data_path /home/s1852803/unmt/XLM/data/processed/en-gu/ \
    --lgs 'en-gu' \
    --mt_steps "en-gu,gu-en" \
    --bt_steps 'en-gu-en,gu-en-gu' \
    --encoder_only false \
    --emb_dim 512 \
    --n_layers 6 \
    --n_heads 8 \
    --dropout 0.1 \
    --attention_dropout 0.1 \
    --gelu_activation true \
    --tokens_per_batch 1000 \
    --batch_size 16 \
    --bptt 128 \
    --optimizer adam_inverse_sqrt,beta1=0.9,beta2=0.98,lr=0.0001 \
    --epoch_size 200000 \
    --eval_bleu true \
    --stopping_criterion valid_en-gu_mt_bleu,10 \
    --validation_metrics valid_en-gu_mt_bleu
