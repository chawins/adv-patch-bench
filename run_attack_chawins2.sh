#!/bin/bash
GPU=1
PATCH_NAME=8_20x20
EXP=47
MODEL_PATH=/data/shared/adv-patch-bench/yolov5/runs/train/exp3/weights/best.pt
CSV_PATH=mapillary_vistas_final_merged.csv
SYN_OBJ_PATH=attack_assets/octagon-915.0.png
OBJ_CLASS=8

CUDA_VISIBLE_DEVICES=$GPU python -u generate_adv_patch.py \
    --device $GPU --seed 0 --data mapillary_vistas.yaml \
    --weights $MODEL_PATH --patch-name $PATCH_NAME --csv-path $CSV_PATH \
    --bg-dir /data/shared/mtsd_v2_fully_annotated/train \
    --save-images --attack-config-path attack_config.yaml \
    --obj-class $OBJ_CLASS --obj-path $SYN_OBJ_PATH \
    --imgsz 2560 --padded_imgsz 1952,2592 \
    --obj-size 128 --num-bg 50 --generate-patch real
    # --imgsz 1280 --padded_imgsz 992,1312 \

CUDA_VISIBLE_DEVICES=$GPU python -u val_attack_synthetic.py \
    --data mapillary_vistas.yaml --tgt-csv-filepath $CSV_PATH \
    --exist-ok --workers 6 --task train --save-exp-metrics \
    --metrics-confidence-threshold 0.359 \
    --adv-patch-path ./runs/val/exp$EXP/$PATCH_NAME.pkl \
    --attack-config-path attack_config.yaml --name $PATCH_NAME \
    --weights $MODEL_PATH --syn-obj-path $SYN_OBJ_PATH \
    --obj-class $OBJ_CLASS --plot-class-examples $OBJ_CLASS \
    --imgsz 2560 --padded_imgsz 1952,2592 --batch-size 2 \
    --interp bilinear --attack-type per-sign
    # --adv-patch-path ./runs/val/exp2/stop_sign_10x10.pkl \
    # --imgsz 1280 --padded_imgsz 992,1312 --batch-size 2 \