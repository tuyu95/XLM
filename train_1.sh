#!/bin/sh

#SBATCH -N 1	  # nodes requested
#SBATCH -n 1	  # tasks requested
#SBATCH --partition=Standard
#SBATCH --gres=gpu:8
#SBATCH --mem=12000  # memory in Mb
#SBATCH --time=0-08:00:00

export CUDA_HOME=/opt/cuda-9.0.176.1/

export CUDNN_HOME=/opt/cuDNN-7.0/

export STUDENT_ID=$(whoami)

export LD_LIBRARY_PATH=${CUDNN_HOME}/lib64:${CUDA_HOME}/lib64:$LD_LIBRARY_PATH

export LIBRARY_PATH=${CUDNN_HOME}/lib64:$LIBRARY_PATH

export CPATH=${CUDNN_HOME}/include:$CPATH

export PATH=${CUDA_HOME}/bin:${PATH}

export PYTHON_PATH=$PATH

source /home/${STUDENT_ID}/miniconda3/bin/activate mlp

devices=0,1,2,3,4,5,6,7

# python -m torch.distributed.launch --nproc_per_node=$NGPU /home/s1852803/unmt/XLM/train.py \

#export NGPU=8; python -m torch.distributed.launch --nproc_per_node=$NGPU /home/s1852803/unmt/XLM/train.py \
CUDA_VISIBLE_DEVICES=$devices python3 /home/s1852803/unmt/XLM/train.py \
    --exp_name unsupMT_enfr \
    --dump_path /home/s1852803/unmt/XLM/dumped/ \
    --reload_model /home/s1852803/unmt/XLM/mlm_enfr_ppl.pth,/home/s1852803/unmt/XLM/mlm_enfr_ppl.pth \
    --data_path /home/s1852803/unmt/XLM/data/processed_1/en-fr/ \
    --lgs 'en-fr' \
    --ae_steps 'en,fr' \
    --bt_steps 'en-fr-en,fr-en-fr' \
    --word_shuffle 3 \
    --word_dropout 0.1 \
    --word_blank 0.1 \
    --lambda_ae '0:1,100000:0.1,300000:0' \
    --encoder_only false \
    --emb_dim 512 \
    --n_layers 6 \
    --n_heads 8 \
    --dropout 0.1 \
    --attention_dropout 0.1 \
    --gelu_activation true \
    --tokens_per_batch 1000 \
    --batch_size 32 \
    --bptt 128 \
    --optimizer adam_inverse_sqrt,beta1=0.9,beta2=0.98,lr=0.0001 \
    --epoch_size 200000 \
    --eval_bleu true \
    --stopping_criterion valid_en-fr_mt_bleu,10 \
    --validation_metrics valid_en-fr_mt_bleu
