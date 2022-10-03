#!/bin/bash

# Detector test script
GPU=1
NUM_GPU=1

BATCH_SIZE=8

# Dataset and model params
DATASET=mapillary-combined-no_color
MODEL_PATH=~/data/adv-patch-bench/yolov5/runs/train/exp11/weights/best.pt

CONF_THRES=0.403
OUTPUT_PATH=./run/val/

# Attack params
CSV_PATH=mapillary_vistas_final_merged.csv
BG_PATH=~/data/mtsd_v2_fully_annotated/test/
IMG_SIZE=1536,2048 # sizes: (1536,2048), (3040,4032)
YOLO_IMG_SIZE=2016
INTERP=bilinear


shapes_arr=("circle-750.0" "triangle-900.0" "triangle_inverted-1220.0" "diamond-600.0" "diamond-915.0" "square-600.0" "rect-458.0-610.0" "rect-762.0-915.0" "rect-915.0-1220.0" "pentagon-915.0" "octagon-915.0")


# mask_name_arr=("10x10" "10x20" "2_10x20")
mask_name_arr=("10x20")
lmbd_arr=(0.00001)
patch_dim_arr=(64)


# Generate Adversarial Patches
for LMBD in ${lmbd_arr[@]};
do
    for PATCH_DIM in ${patch_dim_arr[@]};
    do
        ATTACK_CONFIG_PATH="./configs/attack_config_synthetic_lmbd_${LMBD}_patch_dim_${PATCH_DIM}.yaml"

        for MASK_NAME in ${mask_name_arr[@]};
        do
            SYN_PATCH_PATH="runs/paper_results_new/syn_patches_${LMBD}_patch_dim_${PATCH_DIM}_mask_name_${MASK_NAME}"
            EXP_NAME="synthetic-${MASK_NAME}_bottom"

            for index in "${!shapes_arr[@]}";
            do
                echo "$index -> ${shapes_arr[$index]}"
                OBJ_CLASS=$index

                # Generate adversarial patch
                CUDA_VISIBLE_DEVICES=$GPU python -u gen_patch_yolo.py \
                    --device $GPU --dataset $DATASET --padded-imgsz $IMG_SIZE --name $EXP_NAME \
                    --weights $MODEL_PATH --workers 8 --plot-class-examples $OBJ_CLASS \
                    --tgt-csv-filepath $CSV_PATH --attack-config-path $ATTACK_CONFIG_PATH \
                    --obj-class $OBJ_CLASS --bg-dir $BG_PATH --transform-mode perspective \
                    --interp $INTERP --verbose --synthetic \
                    --obj-size 64 \
                    --imgsz $YOLO_IMG_SIZE \
                    --mask-name $MASK_NAME \
                    --project $SYN_PATCH_PATH
            done
        done
    done
done
        
# Test adversarial patches
# transform: tl + rt
# relight: 0	
for LMBD in ${lmbd_arr[@]};
do
    for PATCH_DIM in ${patch_dim_arr[@]};
    do 
        for MASK_NAME in ${mask_name_arr[@]};
        do
            # path where patch is stored
            echo $SYN_PATCH_PATH

            SYN_PATCH_PATH="runs/paper_results_new/syn_patches_${LMBD}_patch_dim_${PATCH_DIM}_mask_name_${MASK_NAME}"
            EXP_NAME="synthetic-${MASK_NAME}_bottom"

            CUR_EXP_PATH="runs/paper_results_new/syn_attack_syn_gen_lambda_${LMBD}_patch_dim_${PATCH_DIM}_mask_name_${MASK_NAME}"

            for index in "${!shapes_arr[@]}";
            do
                echo "$index -> ${shapes_arr[$index]}"

                OBJ_CLASS=$index
                SHAPE=${shapes_arr[$index]}

                CUDA_VISIBLE_DEVICES=$GPU python -u test_yolo.py \
                    --device $GPU --interp $INTERP --save-exp-metrics --workers 2 \
                    --dataset $DATASET --padded-imgsz $IMG_SIZE --weights $MODEL_PATH \
                    --tgt-csv-filepath $CSV_PATH --attack-config-path $ATTACK_CONFIG_PATH \
                    --name $EXP_NAME --obj-class $OBJ_CLASS --plot-class-examples $OBJ_CLASS \
                    --conf-thres $CONF_THRES --metrics-confidence-threshold $CONF_THRES \
                    --annotated-signs-only --batch-size $BATCH_SIZE --attack-type load \
                    --obj-size 64 \
                    --imgsz $YOLO_IMG_SIZE --transform-mode perspective --verbose --debug --synthetic \
                    --adv-patch-path $SYN_PATCH_PATH/$EXP_NAME/$SHAPE/adv_patch.pkl \
                    --project $CUR_EXP_PATH

            done
        done
    done
done