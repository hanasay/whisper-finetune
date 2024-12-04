#!/bin/bash
full_param=true
run_name="wandb-run-name"

python finetune.py \
    --full_param ${full_param} \
    --timestamps False \
    --base_model openai/whisper-small \
    --train_data dataset/train.jsonl \
    --test_data dataset/eval.jsonl \
    --use_adalora False \
    --fp16 True \
    --use_8bit False \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --gradient_accumulation_steps 1 \
    --num_train_epochs 4 \
    --num_workers 8 \
    --logging_steps 10 \
    --eval_steps 1000 \
    --save_steps 1000 \
    --output_dir output/${run_name} \
    --wandb_run_name ${run_name}
