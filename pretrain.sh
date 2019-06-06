#!/usr/bin/env sh
# Distributed under MIT license


#SBATCH -N 1	  # nodes requested
#SBATCH -n 1	  # tasks requested
#SBATCH --partition=LongJobs
#SBATCH --gres=gpu:4
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

python train.py

## main parameters
--exp_name test_guen_mlm                # experiment name
--dump_path ./dumped/                   # where to store the experiment

## data location / training objective
--data_path ./data/processed/gu-en/     # data location
--lgs 'gu-en'                           # considered languages
--clm_steps ''                          # CLM objective
--mlm_steps 'gu,en'                     # MLM objective

## transformer parameters
--emb_dim 1024                          # embeddings / model dimension
--n_layers 6                            # number of layers
--n_heads 8                             # number of heads
--dropout 0.1                           # dropout
--attention_dropout 0.1                 # attention dropout
--gelu_activation true                  # GELU instead of ReLU

## optimization
--batch_size 32                         # sequences per batch
--bptt 256                              # sequences length
--optimizer adam,lr=0.0001              # optimizer
--epoch_size 200000                     # number of sentences per epoch
--validation_metrics _valid_mlm_ppl     # validation metric (when to save the best model)
--stopping_criterion _valid_mlm_ppl,10  # end experiment if stopping criterion does not improve
