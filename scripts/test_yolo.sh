# Detector test script
GPU=0
NUM_GPU=1

# Dataset and model params
DATASET=mapillary-combined-no_color
MODEL_PATH=./yolov5/runs/train/exp11/weights/best.pt
CONF_THRES=0.571
OUTPUT_PATH=./run/val/

# Attack params
ATTACK_CONFIG_PATH=./configs/attack_config2.yaml
CSV_PATH=mapillary_vistas_final_merged.csv
BG_PATH=~/data/mtsd_v2_fully_annotated/test/
IMG_SIZE=1536,2048 # sizes: (1536,2048), (3040,4032)
YOLO_IMG_SIZE=2016
INTERP=bilinear
OBJ_CLASS=0
# EXP_NAME=none
# EXP_NAME=real_10x10_bottom
EXP_NAME=synthetic-10x10_bottom
# EXP_NAME=per_sign-2_10x20
# EXP_NAME=debug

# Test a detector on Detectron2 without attack
# CUDA_VISIBLE_DEVICES=$GPU python -u test_yolo.py \
#     --device $GPU --interp $INTERP --save-exp-metrics --workers 8 \
#     --dataset $DATASET --padded-imgsz $IMG_SIZE --weights $MODEL_PATH \
#     --tgt-csv-filepath $CSV_PATH --name $EXP_NAME --annotated-signs-only \
#     --metrics-confidence-threshold $CONF_THRES --batch-size 16 --attack-type none

# Generate mask for adversarial patch
# CUDA_VISIBLE_DEVICES=$GPU python -u gen_mask.py \
#     --dataset $DATASET --patch-name $EXP_NAME --obj-class $OBJ_CLASS --obj-size 256 --save-mask

# Test real attack
# CUDA_VISIBLE_DEVICES=$GPU python -u gen_patch_yolo.py \
#     --device $GPU --dataset $DATASET --padded-imgsz $IMG_SIZE --name $EXP_NAME \
#     --weights $MODEL_PATH --workers 6 --plot-class-examples $OBJ_CLASS \
#     --tgt-csv-filepath $CSV_PATH --attack-config-path $ATTACK_CONFIG_PATH \
#     --obj-class $OBJ_CLASS --bg-dir $BG_PATH --transform-mode perspective \
#     --interp $INTERP --verbose --obj-size 256 --imgsz $YOLO_IMG_SIZE

# CUDA_VISIBLE_DEVICES=$GPU python -u test_yolo.py \
#     --device $GPU --interp $INTERP --save-exp-metrics --workers 8 \
#     --dataset $DATASET --padded-imgsz $IMG_SIZE --weights $MODEL_PATH \
#     --tgt-csv-filepath $CSV_PATH --attack-config-path $ATTACK_CONFIG_PATH \
#     --name $EXP_NAME --obj-class $OBJ_CLASS --plot-class-examples $OBJ_CLASS \
#     --conf-thres $CONF_THRES --metrics-confidence-threshold $CONF_THRES \
#     --annotated-signs-only --batch-size 4 --attack-type load \
#     --imgsz $YOLO_IMG_SIZE --transform-mode perspective --verbose

# Test synthetic attack
# Generate adversarial patch (add --synthetic for synthetic attack)
# CUDA_VISIBLE_DEVICES=$GPU python -u gen_patch_yolo.py \
#     --device $GPU --dataset $DATASET --padded-imgsz $IMG_SIZE --name $EXP_NAME \
#     --weights $MODEL_PATH --workers 6 --plot-class-examples $OBJ_CLASS \
#     --tgt-csv-filepath $CSV_PATH --attack-config-path $ATTACK_CONFIG_PATH \
#     --obj-class $OBJ_CLASS --bg-dir $BG_PATH --transform-mode perspective \
#     --interp $INTERP --verbose --synthetic --obj-size 256 --imgsz $YOLO_IMG_SIZE

# Test the generated patch
CUDA_VISIBLE_DEVICES=$GPU python -u test_yolo.py \
    --device $GPU --interp $INTERP --save-exp-metrics --workers 6 \
    --dataset $DATASET --padded-imgsz $IMG_SIZE --weights $MODEL_PATH \
    --tgt-csv-filepath $CSV_PATH --attack-config-path $ATTACK_CONFIG_PATH \
    --name $EXP_NAME --obj-class $OBJ_CLASS --plot-class-examples $OBJ_CLASS \
    --conf-thres $CONF_THRES --metrics-confidence-threshold $CONF_THRES \
    --annotated-signs-only --batch-size 4 --attack-type load \
    --imgsz $YOLO_IMG_SIZE --transform-mode perspective --verbose --synthetic
