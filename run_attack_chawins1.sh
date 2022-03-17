#!/bin/bash
PATCH_NAME=14_synthetic_10x10
EXP=44
GPU=0
MODEL_PATH=/data/shared/adv-patch-bench/yolov5/runs/train/exp2/weights/best.pt
CSV_PATH=mapillary_vistas_final_merged.csv

# CUDA_VISIBLE_DEVICES=$GPU python -u generate_adv_patch.py \
#     --device $GPU --seed 0 --data mapillary_vistas.yaml \
#     --weights $MODEL_PATH --patch-name $PATCH_NAME --csv-path CSV_PATH \
#     --bg-dir /data/shared/mtsd_v2_fully_annotated/train \
#     --save-images --attack-config-path attack_config.yaml \
#     --obj-class 14 --obj-path attack_assets/octagon-915.0.png \
#     --imgsz 1280 --padded_imgsz 992,1312 \
#     --obj-size 128 --num-bg 50 --generate-patch synthetic
    # --imgsz 2560 --padded_imgsz 1952,2592 \

CUDA_VISIBLE_DEVICES=$GPU python -u val_attack_synthetic.py \
    --data mapillary_vistas.yaml \
    --exist-ok --workers 6 --task train --save-exp-metrics \
    --metrics-confidence-threshold 0.359 \
    --adv-patch-path ./runs/val/exp$EXP/$PATCH_NAME.pkl \
    --tgt-csv-filepath mapillary_vistas_final_merged.csv \
    --attack-config-path attack_config.yaml --name $PATCH_NAME \
    --weights $MODEL_PATH \
    --obj-class 14 --plot-class-examples 14 --syn-obj-path attack_assets/octagon-915.0.png \
    --imgsz 1280 --padded_imgsz 992,1312 --batch-size 2 \
    --interp bilinear --attack-type load --synthetic-eval --debug
    # --adv-patch-path ./runs/val/exp2/stop_sign_10x10.pkl \
    # --imgsz 2560 --padded_imgsz 1952,2592 --batch-size 2 \