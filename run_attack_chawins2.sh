#!/bin/bash
PATCH_NAME=14_per-sign_10x10_os256
EXP=25
GPU=1

# CUDA_VISIBLE_DEVICES=$GPU python -u generate_adv_patch.py \
#     --device $GPU \
#     --seed 0 \
#     --data mapillary_vistas.yaml \
#     --weights /data/shared/adv-patch-bench/yolov5/runs/train/exp3/weights/best.pt \
#     --patch-name $PATCH_NAME \
#     --csv-path mapillary_vistas_final_merged.csv \
#     --obj-class 14 \
#     --obj-size 256 \
#     --obj-path attack_assets/octagon-915.0.png \
#     --num-bg 50 \
#     --bg-dir /data/shared/mtsd_v2_fully_annotated/train \
#     --save-images \
#     --generate-patch synthetic \
#     --attack-config-path attack_config.yaml \
#     --imgsz 2560 \
#     --padded_imgsz 1952,2592

CUDA_VISIBLE_DEVICES=$GPU python -u val_attack_synthetic.py \
    --data mapillary_vistas.yaml \
    --exist-ok --workers 6 --task train --save-exp-metrics \
    --metrics-confidence-threshold 0.359 \
    --adv-patch-path ./runs/val/exp$EXP/$PATCH_NAME.pkl \
    --tgt-csv-filepath mapillary_vistas_final_merged.csv \
    --attack-config-path attack_config.yaml --name $PATCH_NAME \
    --weights /data/shared/adv-patch-bench/yolov5/runs/train/exp3/weights/best.pt \
    --obj-class 14 --plot-class-examples 14 --attack-type per-sign  \
    --imgsz 2560 --interp bicubic --padded_imgsz 1952,2592 --batch-size 2
