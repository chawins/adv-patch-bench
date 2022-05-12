#!/bin/bash
GPU=0
EXP=1
MODEL_PATH=~/data/adv-patch-bench/yolov5/runs/train/exp11/weights/best.pt
CSV_PATH=mapillary_vistas_final_merged.csv
SYN_OBJ_PATH=attack_assets/octagon-915.0.png
OBJ_CLASS=10
YOLO_IMG_SIZE=2016
IMG_SIZE=1536,2048 # sizes: (1536,2048), (3040,4032)

# test yolo on real dataset WITHOUT patch
CUDA_VISIBLE_DEVICES=$GPU python -u test_yolo.py \
    --data mapillary_no_color.yaml --dataset mapillary-combined-no_color --task test \
    --tgt-csv-filepath $CSV_PATH --save-exp-metrics \
    --weights $MODEL_PATH --exist-ok --workers 8 \
    --attack-config-path attack_config.yaml --name exp10 \
    --obj-class $OBJ_CLASS --plot-class-examples $OBJ_CLASS --syn-obj-path $SYN_OBJ_PATH \
    --imgsz $YOLO_IMG_SIZE --padded-imgsz $IMG_SIZE --batch-size 16 \
    --attack-type none

PATCH_NAME=10x10_bottom_with_transform
# Generate mask for adversarial patch WITH tranforms
CUDA_VISIBLE_DEVICES=$GPU python -u gen_mask.py \
    --syn-obj-path $SYN_OBJ_PATH --dataset mapillary-combined-no_color \
    --patch-name $PATCH_NAME --obj-class $OBJ_CLASS --obj-size 256 --save-mask

# generate patch WITH tranforms
CUDA_VISIBLE_DEVICES=$GPU python -u gen_patch_yolo.py \
    --device $GPU --seed 0 \
    --data mapillary_no_color.yaml --dataset mapillary-combined-no_color \
    --weights $MODEL_PATH --name $PATCH_NAME --tgt-csv-filepath $CSV_PATH \
    --bg-dir ~/data/yolo_data/images/train \
    --save-images --attack-config-path attack_config.yaml \
    --imgsz $YOLO_IMG_SIZE --padded-imgsz $IMG_SIZE \
    --obj-class $OBJ_CLASS --syn-obj-path $SYN_OBJ_PATH \
    --obj-size 256 --num-bg 10 --attack-type real --interp bilinear

# SAVE_EXP_PATH="${PATCH_NAME}_real_transform"
# echo $SAVE_EXP_PATH

# test patch on real dataset WITH transforms
CUDA_VISIBLE_DEVICES=$GPU python -u test_yolo.py \
    --data mapillary_no_color.yaml --dataset mapillary-combined-no_color --task test \
    --tgt-csv-filepath $CSV_PATH --save-exp-metrics \
    --weights $MODEL_PATH --exist-ok --workers 8 \
    --attack-config-path attack_config.yaml --name $PATCH_NAME \
    --obj-class $OBJ_CLASS --plot-class-examples $OBJ_CLASS --syn-obj-path $SYN_OBJ_PATH \
    --imgsz $YOLO_IMG_SIZE --padded-imgsz $IMG_SIZE --batch-size 16 \
    --metrics-confidence-threshold 0.571 \
    --attack-type load --adv-patch-path ./runs/val/$PATCH_NAME/octagon-915.0/adv_patch.pkl --interp bilinear

PATCH_NAME=10x10_bottom_without_transform
# Generate mask for adversarial patch WITH tranforms
CUDA_VISIBLE_DEVICES=$GPU python -u gen_mask.py \
    --syn-obj-path $SYN_OBJ_PATH --dataset mapillary-combined-no_color \
    --patch-name $PATCH_NAME --obj-class $OBJ_CLASS --obj-size 256 --save-mask

# # generate patch WITHOUT tranforms
CUDA_VISIBLE_DEVICES=$GPU python -u gen_patch_yolo.py \
    --device $GPU --seed 0 \
    --data mapillary_no_color.yaml --dataset mapillary-combined-no_color \
    --weights $MODEL_PATH --name $PATCH_NAME --tgt-csv-filepath $CSV_PATH \
    --bg-dir ~/data/yolo_data/images/train \
    --save-images --attack-config-path attack_config_no_transform.yaml \
    --imgsz $YOLO_IMG_SIZE --padded-imgsz $IMG_SIZE \
    --obj-class $OBJ_CLASS --syn-obj-path $SYN_OBJ_PATH \
    --obj-size 256 --num-bg 56 --attack-type real --interp bilinear

# SAVE_EXP_PATH="${PATCH_NAME}_no_transform"
# test patch on real dataset WITHOUT transforms
CUDA_VISIBLE_DEVICES=$GPU python -u test_yolo.py \
    --data mapillary_no_color.yaml --dataset mapillary-combined-no_color --task test \
    --tgt-csv-filepath $CSV_PATH --save-exp-metrics \
    --weights $MODEL_PATH --exist-ok --workers 8 \
    --attack-config-path attack_config_no_transform.yaml --name $PATCH_NAME \
    --obj-class $OBJ_CLASS --plot-class-examples $OBJ_CLASS --syn-obj-path $SYN_OBJ_PATH \
    --imgsz $YOLO_IMG_SIZE --padded-imgsz $IMG_SIZE --batch-size 16 \
    --metrics-confidence-threshold 0.571 \
    --attack-type load --adv-patch-path ./runs/val/$PATCH_NAME/octagon-915.0/adv_patch.pkl --interp bilinear \
    --no-patch-transform --no-patch-relight