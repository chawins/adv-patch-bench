#!/bin/bash
GPU=3
PATCH_NAME=test-min-area-1600
EXP=49
MODEL_PATH=/data/shared/adv-patch-bench/yolov5/runs/train/exp6/weights/best.pt
CSV_PATH=mapillary_vistas_training_final_merged.csv
SYN_OBJ_PATH=attack_assets/octagon-915.0.png
OBJ_CLASS=14

# CUDA_VISIBLE_DEVICES=$GPU python -u generate_adv_patch.py \
#     --device $GPU --seed 0 --data mapillary_vistas.yaml \
#     --weights $MODEL_PATH --patch-name $PATCH_NAME --csv-path $CSV_PATH \
#     --bg-dir /data/shared/mtsd_v2_fully_annotated/train \
#     --save-images --attack-config-path attack_config.yaml \
#     --obj-class 14 --obj-path attack_assets/octagon-915.0.png \
#     --imgsz 1280 --padded_imgsz 992,1312 \
#     --obj-size 128 --num-bg 50 --generate-patch synthetic
# --imgsz 2560 --padded_imgsz 1952,2592 \

CUDA_VISIBLE_DEVICES=$GPU python -u val_attack_synthetic.py \
    --data mapillary_vistas.yaml --tgt-csv-filepath $CSV_PATH \
    --exist-ok --workers 6 --task train --save-exp-metrics \
    --adv-patch-path ./runs/val/exp$EXP/$PATCH_NAME.pkl \
    --tgt-csv-filepath $CSV_PATH --weights $MODEL_PATH \
    --attack-config-path attack_config.yaml --name $PATCH_NAME \
    --obj-class $OBJ_CLASS --plot-class-examples $OBJ_CLASS --syn-obj-path $SYN_OBJ_PATH \
    --imgsz 2560 --padded_imgsz 1952,2592 --batch-size 4 \
    --interp bilinear --attack-type none --min-area 1600 \
    --metrics-confidence-threshold 0.527
# --data mtsd.yaml --tgt-csv-filepath $CSV_PATH \
# --adv-patch-path ./runs/val/exp2/stop_sign_10x10.pkl \
# --imgsz 1280 --padded_imgsz 992,1312 --batch-size 12
