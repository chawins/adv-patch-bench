# Detector test script
GPU=1
NUM_GPU=1
NUM_WORKERS=4

# Dataset and model params
DATASET=mapillary-combined-no_color
NUM_CLASSES=12
# MODEL=faster_rcnn_R_50_FPN_mtsd_color_1
MODEL=faster_rcnn_R_50_FPN_mtsd_no_color_2
CONF_THRES=0.792
OUTPUT_PATH=~/adv-patch-bench/detectron_output/$MODEL
DETECTRON_CONFIG_PATH=./configs/faster_rcnn_R_50_FPN_3x.yaml

# Attack params
ATTACK_CONFIG_PATH=./configs/attack_config2.yaml
CSV_PATH=mapillary_vistas_final_merged.csv
BG_PATH=~/data/mtsd_v2_fully_annotated/test/
BG_FILES=bg_filenames_octagon-915.0.txt
# BG_FILES=synthetic-10x20/octagon-915.0/bg_filenames-1.txt
IMG_SIZE=1536,2048 # sizes: (1536,2048), (3040,4032)
INTERP=bilinear
SYN_OBJ_SIZE=128
# OBJ_CLASS=10
# EXP_NAME=none
EXP_NAME=real-10x10_bottom
# EXP_NAME=synthetic-10x20
# EXP_NAME=debug
NUM_TEST_SYN=5000

# MODEL.ROI_HEADS.NUM_CLASSES 11(12), 15(16), 401  # This should match model not data
# MODEL.ROI_HEADS.IOU_THRESHOLDS = [0.5]
# MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.5

# Test a detector on Detectron2 without attack
python -u test_detectron.py \
    --num-gpus $NUM_GPU --config-file $DETECTRON_CONFIG_PATH --name no_patch \
    --padded-imgsz $IMG_SIZE --tgt-csv-filepath $CSV_PATH --dataset $DATASET \
    --attack-config-path $ATTACK_CONFIG_PATH --eval-mode drop \
    MODEL.ROI_HEADS.NUM_CLASSES $NUM_CLASSES \
    OUTPUT_DIR $OUTPUT_PATH \
    MODEL.WEIGHTS $OUTPUT_PATH/model_best.pth \
    DATALOADER.NUM_WORKERS $NUM_WORKERS
# Other options
# --debug --resume --annotated-signs-only
# --dataset mtsd-val-no_color, mapillary-combined-no_color

# Generate mask for adversarial patch
# python -u gen_mask.py \
#     --dataset $DATASET --patch-name $EXP_NAME --obj-class $OBJ_CLASS --obj-size 256 --save-mask

# Generate adversarial patch (add --synthetic for synthetic attack)
# python -u gen_patch_detectron.py \
#     --num-gpus $NUM_GPU --config-file $DETECTRON_CONFIG_PATH --interp $INTERP \
#     --dataset $DATASET --padded-imgsz $IMG_SIZE --tgt-csv-filepath $CSV_PATH \
#     --attack-config-path $ATTACK_CONFIG_PATH --obj-class $OBJ_CLASS \
#     --name $EXP_NAME --bg-dir $BG_PATH --transform-mode perspective --verbose \
#     MODEL.ROI_HEADS.NUM_CLASSES $NUM_CLASSES \
#     OUTPUT_DIR $OUTPUT_PATH \
#     MODEL.WEIGHTS $OUTPUT_PATH/model_best.pth \
#     DATALOADER.NUM_WORKERS $NUM_WORKERS

# Test the generated patch
# python -u test_detectron.py \
#     --num-gpus $NUM_GPU --config-file $DETECTRON_CONFIG_PATH --interp $INTERP \
#     --dataset $DATASET --padded-imgsz $IMG_SIZE --eval-mode drop \
#     --tgt-csv-filepath $CSV_PATH --attack-config-path $ATTACK_CONFIG_PATH \
#     --name $EXP_NAME --obj-class $OBJ_CLASS --conf-thres $CONF_THRES \
#     --annotated-signs-only --transform-mode perspective --attack-type debug --debug \
#     MODEL.ROI_HEADS.NUM_CLASSES $NUM_CLASSES \
#     OUTPUT_DIR $OUTPUT_PATH \
#     MODEL.WEIGHTS $OUTPUT_PATH/model_best.pth \
#     DATALOADER.NUM_WORKERS $NUM_WORKERS

# EXP_NAME=synthetic-10x10_bottom

# Generate adversarial patch (add --synthetic for synthetic attack)
# python -u gen_patch_detectron.py \
#     --num-gpus $NUM_GPU --config-file $DETECTRON_CONFIG_PATH --interp $INTERP \
#     --dataset $DATASET --padded-imgsz $IMG_SIZE --tgt-csv-filepath $CSV_PATH \
#     --attack-config-path $ATTACK_CONFIG_PATH --obj-class $OBJ_CLASS \
#     --name $EXP_NAME --bg-dir $BG_PATH --transform-mode perspective --verbose --synthetic \
#     MODEL.ROI_HEADS.NUM_CLASSES $NUM_CLASSES \
#     OUTPUT_DIR $OUTPUT_PATH \
#     MODEL.WEIGHTS $OUTPUT_PATH/model_best.pth \
#     DATALOADER.NUM_WORKERS $NUM_WORKERS

# Test the generated patch
# python -u test_detectron.py \
#     --num-gpus $NUM_GPU --config-file $DETECTRON_CONFIG_PATH --interp $INTERP \
#     --dataset $DATASET --padded-imgsz $IMG_SIZE --eval-mode drop \
#     --tgt-csv-filepath $CSV_PATH --attack-config-path $ATTACK_CONFIG_PATH \
#     --name $EXP_NAME --obj-class $OBJ_CLASS --conf-thres $CONF_THRES \
#     --annotated-signs-only --transform-mode perspective --attack-type load \
#     --synthetic --debug --verbose \
#     MODEL.ROI_HEADS.NUM_CLASSES $NUM_CLASSES \
#     OUTPUT_DIR $OUTPUT_PATH \
#     MODEL.WEIGHTS $OUTPUT_PATH/model_best.pth \
#     DATALOADER.NUM_WORKERS $NUM_WORKERS
# --img-txt-path bg_filenames_octagon-915.0.txt --synthetic \

# --syn-use-scale --syn-use-colorjitter
# =========================================================================== #

syn_attack() {

    NAME=$1
    MASK_NAME=$2
    OBJ_CLASS=$3
    ATK_CONFIG_PATH=./configs/attack_config$4.yaml

    # Generate adversarial patch
    python -u gen_patch_detectron.py \
        --num-gpus $NUM_GPU --config-file $DETECTRON_CONFIG_PATH --interp $INTERP \
        --dataset $DATASET --padded-imgsz $IMG_SIZE --tgt-csv-filepath $CSV_PATH \
        --attack-config-path $ATK_CONFIG_PATH --obj-class $OBJ_CLASS \
        --name $NAME --bg-dir $BG_PATH --transform-mode perspective --verbose \
        --obj-size $SYN_OBJ_SIZE --save-images --mask-name $MASK_NAME \
        --synthetic --img-txt-path $BG_FILES \
        MODEL.ROI_HEADS.NUM_CLASSES $NUM_CLASSES \
        OUTPUT_DIR $OUTPUT_PATH \
        MODEL.WEIGHTS $OUTPUT_PATH/model_best.pth \
        DATALOADER.NUM_WORKERS $NUM_WORKERS

    # Test patch on synthetic signs
    python -u test_detectron.py \
        --num-gpus $NUM_GPU --config-file $DETECTRON_CONFIG_PATH --interp $INTERP \
        --dataset $DATASET --padded-imgsz $IMG_SIZE --eval-mode drop \
        --tgt-csv-filepath $CSV_PATH --attack-config-path $ATK_CONFIG_PATH \
        --name $NAME --obj-class $OBJ_CLASS --conf-thres $CONF_THRES \
        --transform-mode perspective --attack-type load --num-test $NUM_TEST_SYN \
        --synthetic --obj-size $SYN_OBJ_SIZE --img-txt-path $BG_FILES \
        MODEL.ROI_HEADS.NUM_CLASSES $NUM_CLASSES \
        OUTPUT_DIR $OUTPUT_PATH \
        MODEL.WEIGHTS $OUTPUT_PATH/model_best.pth \
        DATALOADER.NUM_WORKERS $NUM_WORKERS

    # Test patch on real signs
    python -u test_detectron.py \
        --num-gpus $NUM_GPU --config-file $DETECTRON_CONFIG_PATH --interp $INTERP \
        --dataset $DATASET --padded-imgsz $IMG_SIZE --eval-mode drop \
        --tgt-csv-filepath $CSV_PATH --attack-config-path $ATK_CONFIG_PATH \
        --name $NAME --obj-class $OBJ_CLASS --conf-thres $CONF_THRES \
        --annotated-signs-only --transform-mode perspective --attack-type load \
        --img-txt-path $BG_FILES \
        MODEL.ROI_HEADS.NUM_CLASSES $NUM_CLASSES \
        OUTPUT_DIR $OUTPUT_PATH \
        MODEL.WEIGHTS $OUTPUT_PATH/model_best.pth \
        DATALOADER.NUM_WORKERS $NUM_WORKERS
}

# syn_attack synthetic-10x20 10x20 0 2
# syn_attack synthetic-10x20 10x20 1 2
# syn_attack synthetic-10x20 10x20 2 2
# syn_attack synthetic-10x20 10x20 3 2
# syn_attack synthetic-10x20 10x20 4 2
# syn_attack synthetic-10x20 10x20 5 2
# syn_attack synthetic-10x20 10x20 6 2
# syn_attack synthetic-10x20 10x20 7 2
# syn_attack synthetic-10x20 10x20 8 2
# syn_attack synthetic-10x20 10x20 9 2
# syn_attack synthetic-10x20 10x20 10 2

# syn_attack synthetic-10x20-adam-ps128-lmd1e0 10x20 2
# syn_attack synthetic-10x20-adam-ps128-lmd1e-1 10x20 3
# syn_attack synthetic-10x20-adam-ps128-lmd1e-2 10x20 4
# syn_attack synthetic-10x20-adam-ps128-lmd1e-3 10x20 5
# syn_attack synthetic-10x20-adam-ps128-lmd1e-4 10x20 6
# syn_attack synthetic-10x20-adam-ps128-lmd1e-5 10x20 7

# syn_attack synthetic-10x10-adam-ps8 10x10 8
# syn_attack synthetic-10x10-adam-ps16 10x10 9
# syn_attack synthetic-10x10-adam-ps32 10x10 10
# syn_attack synthetic-10x10-adam-ps64 10x10 11
# syn_attack synthetic-10x10-adam-ps128 10x10 12
# syn_attack synthetic-10x10-adam-ps256 10x10 13
# syn_attack synthetic-10x10-adam-ps512 10x10 14
