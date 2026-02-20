#!/bin/bash
export accelerate_config="tool_server/tf_eval/scripts/configs/accelerate_config/deepspeed.yaml"
export CUDA_VISIBLE_DEVICES=1
mkdir -p logs

accelerate launch \
  --main_process_port 29401\
  --config_file "${accelerate_config}" \
  -m tool_server.tf_eval \
  --model qwen \
  --model_args pretrained=Qwen/Qwen2.5-7B-Instruct\
  --task_name rsagent \
  --task_type e2e \
  --verbosity INFO \
  --output_path tool_server/tf_eval/results/eval \
  --batch_size 1 \
  --max_rounds 15 \
  --controller_addr http://0.0.0.0:20001 \
  --save_to_ckpt rsagent=tool_server/tf_eval/results/infer/qwen2_5_7b_I_rsagent_ckpt.jsonl \
  --resume_from_ckpt rsagent=tool_server/tf_eval/results/infer/qwen2_5_7b_I_rsagent_ckpt.jsonl \
  --cache_dir data/gpkgs \
  --save_dir tool_server/tool_workers/tools_output \
