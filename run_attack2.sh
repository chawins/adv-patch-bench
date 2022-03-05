#!/bin/bash

# CUDA_VISIBLE_DEVICES=0 python -u generate_adv_patch.py \
#     --seed 0 --data mapillary_vistas.yaml \
#     --weights /data/shared/adv-patch-bench/yolov5/runs/train/exp2/weights/best.pt \
#     --bg-dir /data/shared/mtsd_v2_fully_annotated/train --num-bg 40 \
#     --csv-path ./mapillary_vistas_final_merged.csv \
#     --imgsz 1280 --padded_imgsz 992,1312 --obj-size 128 \
#     --obj-class 14 --obj-path attack_assets/octagon-915.0.png \
#     --generate-patch real --patch-name stop_sign_real_aug \
#     --save-images

# CUDA_VISIBLE_DEVICES=0 python -u val_attack_synthetic.py \
#     --imgsz 1280 --padded_imgsz 992,1312 --obj-size 128 \
#     --batch-size 8 --data mapillary_vistas.yaml \
#     --weights /data/shared/adv-patch-bench/yolov5/runs/train/exp2/weights/best.pt \
#     --exist-ok --workers 8 --task train \
#     --save-exp-metrics --plot-octagons --metrics-confidence-threshold 0.359 \
#     --apply-patch --load-patch ./runs/val/exp15/stop_sign_transform.pkl \
#     --per-sign-attack

CUDA_VISIBLE_DEVICES=0 python -u val_attack_synthetic.py \
    --imgsz 1280 --padded_imgsz 992,1312 --obj-size 128 \
    --batch-size 8 --data mapillary_vistas.yaml \
    --weights /data/shared/adv-patch-bench/yolov5/runs/train/exp2/weights/best.pt \
    --exist-ok --workers 8 --task train \
    --save-exp-metrics --plot-octagons --metrics-confidence-threshold 0.359 \
    --apply-patch --load-patch ./runs/val/exp29/stop_sign_real_aug.pkl \
    --img-txt-path ./runs/val/exp29/bg_filenames.txt
    # --per-sign-attack