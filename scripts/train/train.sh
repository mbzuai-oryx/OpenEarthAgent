#!/bin/bash
set -e
export PYTHONPATH="$(pwd)"

# =========================
# GPU / CUDA
# =========================
export CUDA_VISIBLE_DEVICES=0,1,2,3
export TORCH_CUDA_ARCH_LIST="8.0"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCH_NCCL_BLOCKING_WAIT=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_DEBUG=WARN

# =========================
# Paths
# =========================
export OUTPUT_DIR="OpenEarthAgent"
export MODEL_PATH="unsloth/Qwen3-4B-Instruct-2507"
export DATA_PATH="data/train.json"

# =========================
# Run config
# =========================
export RUN_NAME="OEA_1"
export SEED=42

# =========================
# Distributed setup
# =========================
GPUS_PER_NODE=4
NNODES=1
NODE_RANK=0
MASTER_ADDR=localhost
MASTER_PORT=6007

DISTRIBUTED_ARGS="
  --nproc_per_node $GPUS_PER_NODE \
  --nnodes $NNODES \
  --node_rank $NODE_RANK \
  --master_addr $MASTER_ADDR \
  --master_port $MASTER_PORT
"

# =========================
# Launch
# =========================
torchrun $DISTRIBUTED_ARGS scripts/train.py \
  --output_dir ${OUTPUT_DIR} \
  --model_name_or_path ${MODEL_PATH} \
  --data_files ${DATA_PATH} \
  --dataset_train_split train \
  --seed ${SEED} \
  --max_seq_length 4096 \
  --learning_rate 2e-5 \
  --lr_scheduler_type cosine \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 4 \
  --logging_steps 10 \
  --warmup_steps 0 \
  --warmup_ratio 0.05 \
  --weight_decay 0.01 \
  --num_train_epochs 1 \
  --max_steps -1 \
  --optim adamw_bnb_8bit \
  --bf16 true \
  --gradient_checkpointing true \
  --report_to "tensorboard" \
  --save_strategy steps \
  --save_steps 100 \
  --save_total_limit 2 


