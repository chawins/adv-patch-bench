#!/bin/bash
TORCHELASTIC_MAX_RESTARTS=0
# CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch \
#     --nproc_per_node=2 \
#     --use_env traffic_sign_classifier.py \
#     --wandb \
#     --dist-url tcp://localhost:10005 \
#     --seed 0 \
#     --full-precision \
#     --print-freq 100 \
#     --data /data/shared/mtsd_v2_fully_annotated/cropped_signs_v6/ \
#     --dataset mtsd \
#     --arch resnet18 \
#     --output-dir /data/chawin/adv-patch-bench/results/4 \
#     --epochs 50 \
#     --batch-size 128 \
#     --lr 1e-2 \
#     --wd 1e-4 \
#     --pretrained \
#     --num-classes 12 \
#     --adv-train none \
#     --experiment clf
#     # --evaluate

CUDA_VISIBLE_DEVICES=0 python example_transforms.py \
    --seed 0 \
    --full-precision \
    --batch-size 256 \
    --data /data/shared/mtsd_v2_fully_annotated/cropped_signs_v6/ \
    --dataset mtsd \
    --arch resnet18 \
    --output-dir /data/chawin/adv-patch-bench/results/ \
    --num-classes 12 \
    --resume /data/chawin/adv-patch-bench/results/4/checkpoint_best.pt
