#!/bin/sh

#SBATCH -N 1	  # nodes requested
#SBATCH -n 1	  # tasks requested
#SBATCH --partition=Standard
#SBATCH --gres=gpu:2
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

devices=0,1

cd /home/s1852803/unmt/XLM

cat data/processed/en-fr/test.en-fr.fr | \
CUDA_VISIBLE_DEVICES=$devices python3 /home/s1852803/unmt/XLM/translate.py --exp_name translate \
--model_path /home/s1852803/unmt/XLM/dumped/unsupMT_enfr/360935/best-valid_en-fr_mt_bleu.pth \
--output_path /home/s1852803/unmt/XLM/output \
--src_lang fr --tgt_lang en --batch_size 16
