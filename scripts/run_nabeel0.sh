#!/bin/bash
GPU=0
PATCH_NAME=10x10_bottom
EXP=1
MODEL_PATH=~/data/adv-patch-bench/yolov5/runs/train/exp10/weights/best.pt
CSV_PATH=mapillary_vistas_final_merged.csv
SYN_OBJ_PATH=attack_assets/octagon-915.0.png
OBJ_CLASS=10

CUDA_VISIBLE_DEVICES=$GPU python -u gen_patch_yolo.py \
    --device $GPU --seed 0 \
    --data mapillary_no_color.yaml --dataset mapillary-combined-no_color \
    --weights $MODEL_PATH --name $PATCH_NAME --tgt-csv-filepath $CSV_PATH \
    --bg-dir ~/data/yolo_data/images/train \
    --save-images --attack-config-path attack_config.yaml \
    --imgsz 2000 --padded-imgsz 1536,2048 \
    --obj-class $OBJ_CLASS --syn-obj-path $SYN_OBJ_PATH \
    --obj-size 256 --num-bg 5 --attack-type real
# --imgsz 2560 --padded_imgsz 1952,2592 \

# --imgsz 1280 --padded_imgsz 960,1280 \

# CUDA_VISIBLE_DEVICES=$GPU python -u val_attack_synthetic.py \
#     --data mapillary_no_color.yaml --tgt-csv-filepath $CSV_PATH \
#     --exist-ok --workers 8 --task test --save-exp-metrics \
#     --adv-patch-path ./runs/val/exp$EXP/$PATCH_NAME.pkl \
#     --tgt-csv-filepath $CSV_PATH --weights $MODEL_PATH \
#     --attack-config-path attack_config.yaml --name $PATCH_NAME \
#     --obj-class $OBJ_CLASS --plot-class-examples $OBJ_CLASS --syn-obj-path $SYN_OBJ_PATH \
#     --imgsz 4000 --padded_imgsz 3000,4000 --batch-size 1 \
#     --interp bilinear --attack-type none --min-area 600 --debug
# --metrics-confidence-threshold 0.527
# --data mtsd.yaml --tgt-csv-filepath $CSV_PATH \
# --adv-patch-path ./runs/val/exp2/stop_sign_10x10.pkl \
# --imgsz 1280 --padded_imgsz 992,1312 --batch-size 12
#  --imgsz 2560 --padded_imgsz 1952,2592 --batch-size 1 \
