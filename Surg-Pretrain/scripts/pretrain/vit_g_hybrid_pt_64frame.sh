#!/usr/bin/env bash
set -x

export MASTER_PORT=$((12000 + $RANDOM % 20000))
export OMP_NUM_THREADS=1

OUTPUT_DIR='./vit_g_hybrid_pt_foreground_mask_0808_frame_64_mask_tube_length_2'
DATA_PATH='/mnt/xingjian_luo/project/VideoMAEv2/cholect50.csv'

JOB_NAME=$1
PARTITION=${PARTITION:-"video"}
# 8 for 1 node, 16 for 2 node, etc.
GPUS=${GPUS:-64}
GPUS_PER_NODE=${GPUS_PER_NODE:-8}
CPUS_PER_TASK=${CPUS_PER_TASK:-12}
SRUN_ARGS=${SRUN_ARGS:-""}
PY_ARGS=${@:2}

# batch_size can be adjusted according to the graphics card
OUTPUT_DIR='./vit_g_hybrid_pt_foreground_mask_0808_frame_64_mask_tube_length_2_4'
python -u run_mae_pretraining.py \
        --data_path ${DATA_PATH} \
        --mask_type tube \
        --mask_ratio 0.8 \
        --tubelet_size 2 \
        --mask_tube_len 4\
        --decoder_mask_type run_cell \
        --decoder_mask_ratio 0.5 \
        --model pretrain_videomae_giant_patch14_224 \
        --resume /mnt/xingjian_luo/project/VideoMAEv2/vit_g_hybrid_pt_foreground_mask_0808_frame_64_mask_tube_length_2/checkpoint-599.pth \
        --mask_path /mnt/xingjian_luo/project/VideoMAEv2/masks/mask.json \
        --decoder_depth 4 \
        --batch_size 1 \
        --with_checkpoint \
        --num_frames 64 \
        --sampling_rate 1 \
        --num_sample 4 \
        --num_workers 1 \
        --opt adamw \
        --lr 5e-4 \
        --clip_grad 0.02 \
        --opt_betas 0.9 0.95 \
        --warmup_epochs 30 \
        --save_ckpt_freq 500 \
        --epochs 1200 \
        --log_dir ${OUTPUT_DIR} \
        --output_dir ${OUTPUT_DIR} \
        ${PY_ARGS}

OUTPUT_DIR='./vit_g_hybrid_pt_foreground_mask_0808_frame_64_mask_tube_length_2_4_8'
python -u run_mae_pretraining.py \
        --data_path ${DATA_PATH} \
        --mask_type tube \
        --mask_ratio 0.8 \
        --tubelet_size 2 \
        --mask_tube_len 8\
        --decoder_mask_type run_cell \
        --decoder_mask_ratio 0.5 \
        --model pretrain_videomae_giant_patch14_224 \
        --resume /mnt/xingjian_luo/project/VideoMAEv2/vit_g_hybrid_pt_foreground_mask_0808_frame_64_mask_tube_length_2_4/checkpoint-1199.pth \
        --mask_path /mnt/xingjian_luo/project/VideoMAEv2/masks/mask.json \
        --decoder_depth 4 \
        --batch_size 1 \
        --with_checkpoint \
        --num_frames 64 \
        --sampling_rate 1 \
        --num_sample 4 \
        --num_workers 1 \
        --opt adamw \
        --lr 5e-4 \
        --clip_grad 0.02 \
        --opt_betas 0.9 0.95 \
        --warmup_epochs 30 \
        --save_ckpt_freq 500 \
        --epochs 1800 \
        --log_dir ${OUTPUT_DIR} \
        --output_dir ${OUTPUT_DIR} \
        ${PY_ARGS}

OUTPUT_DIR='./vit_g_hybrid_pt_foreground_mask_0808_frame_64_mask_tube_length_2_4_8_16'
python -u run_mae_pretraining.py \
        --data_path ${DATA_PATH} \
        --mask_type tube \
        --mask_ratio 0.8 \
        --tubelet_size 2 \
        --mask_tube_len 16\
        --decoder_mask_type run_cell \
        --decoder_mask_ratio 0.5 \
        --model pretrain_videomae_giant_patch14_224 \
        --resume /mnt/xingjian_luo/project/VideoMAEv2/vit_g_hybrid_pt_foreground_mask_0808_frame_64_mask_tube_length_2_4_8/checkpoint-1799.pth \
        --mask_path /mnt/xingjian_luo/project/VideoMAEv2/masks/mask.json \
        --decoder_depth 4 \
        --batch_size 1 \
        --with_checkpoint \
        --num_frames 64 \
        --sampling_rate 1 \
        --num_sample 4 \
        --num_workers 1 \
        --opt adamw \
        --lr 5e-4 \
        --clip_grad 0.02 \
        --opt_betas 0.9 0.95 \
        --warmup_epochs 30 \
        --save_ckpt_freq 500 \
        --epochs 2400 \
        --log_dir ${OUTPUT_DIR} \
        --output_dir ${OUTPUT_DIR} \
        ${PY_ARGS}