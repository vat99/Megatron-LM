#! /bin/bash

# Runs the "345M" parameter model

RANK=0
WORLD_SIZE=1

DATA_PATH=/workspace/Megatron-LM/my-gpt2_text_document

python pretrain_gpt.py \
       --num-layers 48 \
       --hidden-size 1600 \
       --num-attention-heads 25 \
       --micro-batch-size 16 \
       --global-batch-size 64 \
       --seq-length 1024 \
       --max-position-embeddings 1024 \
       --train-iters 1000 \
       --lr-decay-iters 320 \
       --data-path $DATA_PATH \
       --vocab-file /workspace/Megatron-LM/gpt2-vocab.json \
       --merge-file /workspace/Megatron-LM/gpt2-merges.txt \
       --data-impl mmap \
       --split 800,100,100 \
       --distributed-backend nccl \
       --lr 0.00015 \
       --min-lr 1.0e-5 \
       --lr-decay-style cosine \
       --weight-decay 1e-2 \
       --clip-grad 1.0 \
       --lr-warmup-fraction .01 \
       --checkpoint-activations \
       --log-interval 1 \
       --save-interval 10000 \
       --eval-interval 1000 \
       --eval-iters 10 \
       --fp16
