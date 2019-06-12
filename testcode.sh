#!/bin/sh

python3 /home/s1852803/unmt/XLM/train.py \
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
