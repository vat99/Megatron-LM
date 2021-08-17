#! /bin/bash

# Runs the "345M" parameter model

GPUS_PER_NODE=1
# Change for multinode config
MASTER_ADDR=localhost
MASTER_PORT=6000
NNODES=1
NODE_RANK=0
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))

DATA_PATH=/workspace/Megatron-LM/my-gpt2_text_document

DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE --nnodes $NNODES --node_rank $NODE_RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT"

python -m torch.distributed.launch $DISTRIBUTED_ARGS \
       pretrain_gpt.py \
       --num-layers 48 \
       --hidden-size 1600 \
       --num-attention-heads 25 \
       --micro-batch-size 8 \
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
       --lr 0.000015 \
       --lr-decay-style cosine \
       --min-lr 1.0e-5 \
       --weight-decay 1e-2 \
       --clip-grad 1.0 \
       --lr-warmup-fraction .01 \
       --checkpoint-activations \
       --log-interval 1 \
       --save-interval 10000 \
       --eval-interval 1000 \
       --eval-iters 10 \
       --fp16
