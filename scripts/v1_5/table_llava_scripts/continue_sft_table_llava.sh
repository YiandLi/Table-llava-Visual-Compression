#!/bin/bash

# mlp2x_gelu_SVD，mlp2x_gelu
nohup deepspeed llava/train/train_mem.py \
    --deepspeed ./scripts/zero2.json \
    --lora_enable True --bits 4  --lora_r 128 --lora_alpha 256 --mm_projector_lr 2e-5 \
    --model_name_or_path ./model_ckpt/vicuna \
    --pretrain_mm_mlp_adapter ./model_ckpt/mm_projector.bin \
    --vision_tower ./model_ckpt/clip_large_336 \
    --mm_projector_type mlp2x_gelu \
    --version v1 \
    --data_path ./LLaVA-Finetune/enhanced_llava_sft_data_898K.json \
    --image_folder ./LLaVA-Finetune/images \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir ./checkpoints/llava-v1.5-7b-sft-with-table \
    --num_train_epochs 1 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 50000 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2560 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to none | tee ./logs/sft_table_llava_7b_epoch_1_max_length_2560.log