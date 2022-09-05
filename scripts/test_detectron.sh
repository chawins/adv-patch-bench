# Detector test script
GPU=1
NUM_GPU=1
NUM_WORKERS=6

# Dataset and model params
DATASET=mapillary-combined-no_color
MODEL=faster_rcnn_R_50_FPN_mtsd_no_color_2
CONF_THRES=0.792
MODEL_PATH=~/adv-patch-bench/detectron_output/$MODEL/model_best.pth
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
OBJ_CLASS=7
# EXP_NAME=none
EXP_NAME=real-10x10_bottom
# EXP_NAME=synthetic-10x20
# EXP_NAME=debug
NUM_TEST_SYN=5000

# =========================================================================== #

# Test a detector on Detectron2 without attack
# python -u test_detectron.py \
#     --num-gpus $NUM_GPU --config-file $DETECTRON_CONFIG_PATH --name no_patch \
#     --padded-imgsz $IMG_SIZE --tgt-csv-filepath $CSV_PATH --dataset $DATASET \
#     --attack-config-path $ATTACK_CONFIG_PATH --workers $NUM_WORKERS \
#     --weights $MODEL_PATH --eval-mode drop --annotated-signs-only --conf-thres $CONF_THRES

# For synthetic signs, we have to pick one class at a time and there are extra
# args to set
# for i in {0..10}; do
#     python -u test_detectron.py \
#         --num-gpus $NUM_GPU --config-file $DETECTRON_CONFIG_PATH --name no_patch \
#         --padded-imgsz $IMG_SIZE --tgt-csv-filepath $CSV_PATH --dataset $DATASET \
#         --attack-config-path $ATTACK_CONFIG_PATH --workers $NUM_WORKERS \
#         --weights $MODEL_PATH --eval-mode drop --annotated-signs-only \
#         --obj-class $i --obj-size $SYN_OBJ_SIZE --conf-thres $CONF_THRES \
#         --num-test $NUM_TEST_SYN --synthetic
# done
# Other options
# --debug
# --dataset mtsd-val-no_color, mapillary-combined-no_color

# python -u test_detectron.py \
#     --num-gpus $NUM_GPU --config-file $DETECTRON_CONFIG_PATH --name no_patch \
#     --padded-imgsz $IMG_SIZE --tgt-csv-filepath $CSV_PATH --dataset $DATASET \
#     --attack-config-path $ATTACK_CONFIG_PATH --workers $NUM_WORKERS \
#     --weights $MODEL_PATH --eval-mode drop --annotated-signs-only \
#     --obj-class $OBJ_CLASS --obj-size $SYN_OBJ_SIZE --conf-thres $CONF_THRES \
#     --num-test $NUM_TEST_SYN --synthetic

# =========================================================================== #

# Generate mask for adversarial patch
# python -u gen_mask.py \
#     --dataset $DATASET --patch-name $EXP_NAME --obj-class $OBJ_CLASS --obj-size 256 --save-mask

# =========================================================================== #

# Generate adversarial patch (add --synthetic for synthetic attack)
# python -u gen_patch_detectron.py \
#     --num-gpus $NUM_GPU --config-file $DETECTRON_CONFIG_PATH --interp $INTERP \
#     --dataset $DATASET --padded-imgsz $IMG_SIZE --tgt-csv-filepath $CSV_PATH \
#     --attack-config-path $ATTACK_CONFIG_PATH --obj-class $OBJ_CLASS \
#     --name $EXP_NAME --bg-dir $BG_PATH --transform-mode perspective --verbose \
#     --weights $MODEL_PATH --workers $NUM_WORKERS

# =========================================================================== #

# Test the generated patch
# python -u test_detectron.py \
#     --num-gpus $NUM_GPU --config-file $DETECTRON_CONFIG_PATH --interp $INTERP \
#     --dataset $DATASET --padded-imgsz $IMG_SIZE --eval-mode drop \
#     --tgt-csv-filepath $CSV_PATH --attack-config-path $ATTACK_CONFIG_PATH \
#     --name $EXP_NAME --obj-class $OBJ_CLASS --conf-thres $CONF_THRES \
#     --annotated-signs-only --transform-mode perspective --attack-type debug --debug \
#     --weights $MODEL_PATH --workers $NUM_WORKERS

# =========================================================================== #

# EXP_NAME=synthetic-10x10_bottom

# Generate adversarial patch (add --synthetic for synthetic attack)
# python -u gen_patch_detectron.py \
#     --num-gpus $NUM_GPU --config-file $DETECTRON_CONFIG_PATH --interp $INTERP \
#     --dataset $DATASET --padded-imgsz $IMG_SIZE --tgt-csv-filepath $CSV_PATH \
#     --attack-config-path $ATTACK_CONFIG_PATH --obj-class $OBJ_CLASS \
#     --name $EXP_NAME --bg-dir $BG_PATH --transform-mode perspective --verbose --synthetic \
#     --weights $MODEL_PATH --workers $NUM_WORKERS

# Test the generated patch
# python -u test_detectron.py \
#     --num-gpus $NUM_GPU --config-file $DETECTRON_CONFIG_PATH --interp $INTERP \
#     --dataset $DATASET --padded-imgsz $IMG_SIZE --eval-mode drop \
#     --tgt-csv-filepath $CSV_PATH --attack-config-path $ATTACK_CONFIG_PATH \
#     --name $EXP_NAME --obj-class $OBJ_CLASS --conf-thres $CONF_THRES \
#     --annotated-signs-only --transform-mode perspective --attack-type load \
#     --synthetic --debug --verbose \
#     --weights $MODEL_PATH --workers $NUM_WORKERS
# --img-txt-path bg_filenames_octagon-915.0.txt --synthetic \

# --syn-use-scale --syn-use-colorjitter
# =========================================================================== #

syn_attack() {

    NAME=$1
    MASK_NAME=$2
    OBJ_CLASS=$3
    ATK_CONFIG_PATH=./configs/attack_config$4.yaml

    case $OBJ_CLASS in
    0) BG_FILES=bg_filenames_circle-750.0.txt ;;
    1) BG_FILES=bg_filenames_triangle-900.0.txt ;;
    2) BG_FILES=bg_filenames_triangle_inverted-1220.0.txt ;;
    3) BG_FILES=bg_filenames_diamond-600.0.txt ;;
    4) BG_FILES=bg_filenames_diamond-915.0.txt ;;
    5) BG_FILES=bg_filenames_square-600.0.txt ;;
    6) BG_FILES=bg_filenames_rect-458.0-610.0.txt ;;
    7) BG_FILES=bg_filenames_rect-762.0-915.0.txt ;;
    8) BG_FILES=bg_filenames_rect-915.0-1220.0.txt ;;
    9) BG_FILES=bg_filenames_pentagon-915.0.txt ;;
    10) BG_FILES=bg_filenames_octagon-915.0.txt ;;
    esac

    # Generate adversarial patch
    python -u gen_patch_detectron.py \
        --num-gpus $NUM_GPU --config-file $DETECTRON_CONFIG_PATH --interp $INTERP \
        --dataset $DATASET --padded-imgsz $IMG_SIZE --tgt-csv-filepath $CSV_PATH \
        --attack-config-path $ATK_CONFIG_PATH --obj-class $OBJ_CLASS \
        --name $NAME --bg-dir $BG_PATH --transform-mode perspective \
        --weights $MODEL_PATH --workers $NUM_WORKERS --mask-name $MASK_NAME \
        --img-txt-path $BG_FILES \
        --save-images --obj-size $SYN_OBJ_SIZE --synthetic --verbose

    # Test patch on synthetic signs
    python -u test_detectron.py \
        --num-gpus $NUM_GPU --config-file $DETECTRON_CONFIG_PATH --interp $INTERP \
        --dataset $DATASET --padded-imgsz $IMG_SIZE --eval-mode drop \
        --tgt-csv-filepath $CSV_PATH --attack-config-path $ATK_CONFIG_PATH \
        --name $NAME --obj-class $OBJ_CLASS --conf-thres $CONF_THRES \
        --weights $MODEL_PATH --workers $NUM_WORKERS --transform-mode perspective \
        --img-txt-path $BG_FILES \
        --num-test $NUM_TEST_SYN --obj-size $SYN_OBJ_SIZE --attack-type load --synthetic

    # Test patch on real signs
    python -u test_detectron.py \
        --num-gpus $NUM_GPU --config-file $DETECTRON_CONFIG_PATH --interp $INTERP \
        --dataset $DATASET --padded-imgsz $IMG_SIZE --eval-mode drop \
        --tgt-csv-filepath $CSV_PATH --attack-config-path $ATK_CONFIG_PATH \
        --name $NAME --obj-class $OBJ_CLASS --conf-thres $CONF_THRES \
        --weights $MODEL_PATH --workers $NUM_WORKERS --transform-mode perspective \
        --img-txt-path $BG_FILES \
        --attack-type load --annotated-signs-only
}

syn_attack_all() {
    for i in {0..10}; do
        syn_attack synthetic-10x20 10x20 $i 2
    done
}

syn_attack_all

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
