#!/bin/bash

# Detector test script
GPU=1
NUM_GPU=1
NUM_WORKERS=6

# Dataset and model params
DATASET=mapillary-combined-no_color # Options: mapillary-combined-no_color, mtsd-no_color
MODEL=faster_rcnn_R_50_FPN_mtsd_no_color_2
MODEL_PATH=~/adv-patch-bench/detectron_output/$MODEL/model_best.pth
DETECTRON_CONFIG_PATH=./configs/faster_rcnn_R_50_FPN_3x.yaml
CSV_PATH=mapillary_vistas_final_merged.csv
BG_PATH=~/data/mtsd_v2_fully_annotated/test/
CONF_THRES=0.634
IMG_SIZE=1536,2048 # sizes: (1536,2048), (3040,4032)
NUM_TEST_SYN=5000

# Attack params
# MASK_SIZE=10x10
SYN_OBJ_SIZE=64
ATK_CONFIG_PATH=./configs/attack_config_azure2.yaml

INTERP=bilinear
TF_MODE=perspective
# synthetic-10x10-obj64-pd64-ld0.00001-2cj0.05.out
# EXP_NAME=synthetic-${MASK_SIZE}-obj${SYN_OBJ_SIZE}-pd64-ld0.00001-2cj0.05  # TODO: rename
# EXP_NAME=real-${MASK_SIZE}-pd64-ld0.00001-rt15-sc0-var
CLEAN_EXP_NAME=no_patch_syn_${SYN_OBJ_SIZE}_2cj0.05

DATASET=mapillary-combined-no_color
CUDA_VISIBLE_DEVICES=$GPU python -u test_detectron.py \
    --seed 0 --num-gpus $NUM_GPU --config-file $DETECTRON_CONFIG_PATH \
    --interp $INTERP --dataset $DATASET --padded-imgsz $IMG_SIZE --eval-mode drop \
    --tgt-csv-filepath $CSV_PATH --attack-config-path "$ATK_CONFIG_PATH" \
    --name mtsd --obj-class -1 --conf-thres $CONF_THRES --weights $MODEL_PATH \
    --workers $NUM_WORKERS --transform-mode $TF_MODE --attack-type none 

function syn_attack {

    OBJ_CLASS=$1
    MASK_SIZE=$2
    # NUM_BG=200
    EXP_NAME=real-${MASK_SIZE}-pd64-ld0.00001-rt15-sc0

    case $OBJ_CLASS in
    0) OBJ_CLASS_NAME=circle-750.0 ;;
    1) OBJ_CLASS_NAME=triangle-900.0 ;;
    2) OBJ_CLASS_NAME=triangle_inverted-1220.0 ;;
    3) OBJ_CLASS_NAME=diamond-600.0 ;;
    4) OBJ_CLASS_NAME=diamond-915.0 ;;
    5) OBJ_CLASS_NAME=square-600.0 ;;
    6) OBJ_CLASS_NAME=rect-458.0-610.0 ;;
    7) OBJ_CLASS_NAME=rect-762.0-915.0 ;;
    8) OBJ_CLASS_NAME=rect-915.0-1220.0 ;;
    9) OBJ_CLASS_NAME=pentagon-915.0 ;;
    10) OBJ_CLASS_NAME=octagon-915.0 ;;
    esac

    BG_FILES=bg_filenames_"$OBJ_CLASS_NAME".txt

    # CUDA_VISIBLE_DEVICES=$GPU python -u test_detectron.py \
    #     --seed 0 --num-gpus $NUM_GPU --config-file $DETECTRON_CONFIG_PATH --interp $INTERP \
    #     --dataset $DATASET --padded-imgsz $IMG_SIZE --eval-mode drop \
    #     --tgt-csv-filepath $CSV_PATH --attack-config-path "$ATK_CONFIG_PATH" \
    #     --name no_patch --obj-class "$OBJ_CLASS" --conf-thres $CONF_THRES \
    #     --mask-name "$MASK_SIZE" --weights $MODEL_PATH --workers $NUM_WORKERS \
    #     --transform-mode $TF_MODE --img-txt-path $BG_FILES --attack-type none \
    #     --annotated-signs-only &&

    # Test on synthetic clean samples (should only be done once per aug method)
    # CUDA_VISIBLE_DEVICES=$GPU python -u test_detectron.py \
    #     --seed 0 --num-gpus $NUM_GPU --config-file $DETECTRON_CONFIG_PATH --name $CLEAN_EXP_NAME \
    #     --padded-imgsz $IMG_SIZE --tgt-csv-filepath $CSV_PATH --dataset $DATASET \
    #     --attack-config-path "$ATK_CONFIG_PATH" --workers $NUM_WORKERS --interp $INTERP \
    #     --weights $MODEL_PATH --eval-mode drop --annotated-signs-only \
    #     --obj-class "$OBJ_CLASS" --obj-size $SYN_OBJ_SIZE --conf-thres $CONF_THRES \
    #     --img-txt-path $BG_FILES --num-test $NUM_TEST_SYN --synthetic \
    #     --syn-use-colorjitter --syn-colorjitter-intensity 0.05 &&

    # Generate adversarial patch
    CUDA_VISIBLE_DEVICES=$GPU python -u gen_patch_detectron.py \
        --seed 0 --num-gpus $NUM_GPU --config-file $DETECTRON_CONFIG_PATH --interp $INTERP \
        --dataset $DATASET --padded-imgsz $IMG_SIZE --tgt-csv-filepath $CSV_PATH \
        --attack-config-path "$ATK_CONFIG_PATH" --obj-class "$OBJ_CLASS" \
        --name "$EXP_NAME" --bg-dir $BG_PATH --transform-mode $TF_MODE \
        --weights $MODEL_PATH --workers $NUM_WORKERS --mask-name "$MASK_SIZE" \
        --save-images --obj-size $SYN_OBJ_SIZE \
        --annotated-signs-only --verbose --img-txt-path $BG_FILES &&
    # --synthetic 

    # BG_FILES=bg_filenames_"$OBJ_CLASS_NAME"_"$NUM_BG".txt
    # cp detectron_output/"$EXP_NAME"/"$OBJ_CLASS_NAME"/bg_filenames-"$NUM_BG".txt bg_txt_files/$BG_FILES

    # Test patch on synthetic signs
    # CUDA_VISIBLE_DEVICES=$GPU python -u test_detectron.py \
    #     --seed 0 --num-gpus $NUM_GPU --config-file $DETECTRON_CONFIG_PATH --interp $INTERP \
    #     --dataset $DATASET --padded-imgsz $IMG_SIZE --eval-mode drop \
    #     --tgt-csv-filepath $CSV_PATH --attack-config-path "$ATK_CONFIG_PATH" \
    #     --name "$EXP_NAME" --obj-class "$OBJ_CLASS" --conf-thres $CONF_THRES \
    #     --mask-name "$MASK_SIZE" --weights $MODEL_PATH --workers $NUM_WORKERS \
    #     --img-txt-path $BG_FILES --attack-type load --annotated-signs-only \
    #     --synthetic --obj-size $SYN_OBJ_SIZE --num-test $NUM_TEST_SYN \
    #     --syn-use-colorjitter --syn-colorjitter-intensity 0.05 &&

    # Test patch on real signs
    CUDA_VISIBLE_DEVICES=$GPU python -u test_detectron.py \
        --seed 0 --num-gpus $NUM_GPU --config-file $DETECTRON_CONFIG_PATH --interp $INTERP \
        --dataset $DATASET --padded-imgsz $IMG_SIZE --eval-mode drop \
        --tgt-csv-filepath $CSV_PATH --attack-config-path "$ATK_CONFIG_PATH" \
        --name "$EXP_NAME" --obj-class "$OBJ_CLASS" --conf-thres $CONF_THRES \
        --mask-name "$MASK_SIZE" --weights $MODEL_PATH --workers $NUM_WORKERS \
        --transform-mode $TF_MODE --img-txt-path $BG_FILES --attack-type load \
        --annotated-signs-only &&

    echo "Done with $OBJ_CLASS."
}

function syn_attack_all {
    # syn_attack 1 10x10
    # for i in 0 1 2 3 4 5 6 7 8 10; do
    #     syn_attack "$i" 10x10
    # done
    for i in {0..10}; do
        syn_attack "$i" 10x20
    done
    for i in {0..10}; do
        syn_attack "$i" 2_10x20
    done
    # for i in {0..10}; do
    #     syn_attack "$i" 10x4
    # done
    # for i in {0..10}; do
    #     syn_attack "$i" 10x2
    # done
}

# syn_attack_all

exit 0

# =========================================================================== #
#                                Extra Commands                               #
# =========================================================================== #
# Evaluate on all Mapillary Vistas signs
rm ./detectron_output/mapillary_combined_coco_format.json
DATASET=mapillary-combined-no_color
CUDA_VISIBLE_DEVICES=$GPU python -u test_detectron.py \
    --num-gpus $NUM_GPU --config-file $DETECTRON_CONFIG_PATH --name no_patch \
    --padded-imgsz $IMG_SIZE --tgt-csv-filepath $CSV_PATH --dataset $DATASET \
    --attack-config-path $ATK_CONFIG_PATH --workers $NUM_WORKERS \
    --weights $MODEL_PATH --img-txt-path $BG_FILES --eval-mode drop --obj-class -1