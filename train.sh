#!/bin/sh

#SBATCH -N 1	  # nodes requested
#SBATCH -n 1	  # tasks requested
#SBATCH --partition=LongJobs
#SBATCH --gres=gpu:3
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

devices=0,1,2,3

CUDA_VISIBLE_DEVICES=$devices python3 /home/s1852803/unmt/XLM/train.py \
    --exp_name unsupMT_guen                                       # experiment name
    --dump_path ./dumped/                                         # where to store the experiment
    --reload_model 'best-valid_mlm_ppl.pth,best-valid_mlm_ppl.pth'          # model to reload for encoder,decoder
    --data_path ./data/processed/gu-en/                           # data location
    --lgs 'gu-en'                                                 # considered languages
    --ae_steps 'gu,en'                                            # denoising auto-encoder training steps
    --bt_steps 'gu-en-gu,en-gu-en'                                # back-translation steps
    --word_shuffle 3                                              # noise for auto-encoding loss
    --word_dropout 0.1                                            # noise for auto-encoding loss
    --word_blank 0.1                                              # noise for auto-encoding loss
    --lambda_ae '0:1,100000:0.1,300000:0'                         # scheduling on the auto-encoding coefficient
    --encoder_only false                                          # use a decoder for MT
    --emb_dim 1024                                                # embeddings / model dimension
    --n_layers 6                                                  # number of layers
    --n_heads 8                                                   # number of heads
    --dropout 0.1                                                 # dropout
    --attention_dropout 0.1                                       # attention dropout
    --gelu_activation true                                        # GELU instead of ReLU
    --tokens_per_batch 2000                                       # use batches with a fixed number of words
    --batch_size 16                                               # batch size (for back-translation)
    --bptt 128                                                    # sequence length
    --optimizer adam_inverse_sqrt,beta1=0.9,beta2=0.98,lr=0.0001  # optimizer
    --epoch_size 200000                                           # number of sentences per epoch
    --eval_bleu true                                              # also evaluate the BLEU score
    --stopping_criterion 'valid_gu-en_mt_bleu,10'                 # validation metric (when to save the best model)
    --validation_metrics 'valid_gu-en_mt_bleu'                    # end experiment if stopping criterion does not improve
