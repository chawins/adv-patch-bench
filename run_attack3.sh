#!/bin/bash

# CUDA_VISIBLE_DEVICES=0 python -u generate_adv_patch.py \
#     --seed 0 \
#     --data mapillary_vistas.yaml \
#     --weights /data/shared/adv-patch-bench/yolov5/runs/train/exp3/weights/best.pt \
#     --patch-name stop_sign_synthetic_generated \
#     --csv-path mapillary_vistas_final_merged.csv \
#     --obj-class 14 \
#     --obj-size 128 \
#     --obj-path attack_assets/octagon-915.0.png \
#     --num-bg 1 \
#     --bg-dir /data/shared/mtsd_v2_fully_annotated/train \
#     --save-images \
#     --generate-patch real \
#     --attack-config-path attack_config.yaml \
#     --imgsz 2560 \
#     --padded_imgsz 1952,2592

CUDA_VISIBLE_DEVICES=1 python -u val_attack_synthetic.py \
    --imgsz 2560 \
    --padded_imgsz 1952,2592 \
    --batch-size 2 \
    --data mapillary_vistas.yaml \
    --weights /data/shared/adv-patch-bench/yolov5/runs/train/exp3/weights/best.pt \
    --exist-ok \
    --workers 6 \
    --task train \
    --save-exp-metrics \
    --interp bicubic \
    --attack-type per-sign \
    --metrics-confidence-threshold 0.359 \
    --adv-patch-path ./runs/val/exp3/stop_sign_20x20.pkl \
    --tgt-csv-filepath mapillary_vistas_final_merged.csv \
    --attack-config-path attack_config.yaml \
    --obj-class 14 \
    --plot-class-examples 14 \
    --name exp-20x20