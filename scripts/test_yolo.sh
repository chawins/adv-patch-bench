# Detector test script
GPU=0
NUM_GPU=1

# Dataset and model params
DATASET=mapillary-combined-no_color
MODEL_PATH=~/data/adv-patch-bench/yolov5/runs/train/exp11/weights/best.pt
CONF_THRES=0.571
OUTPUT_PATH=./run/val/

# Attack params
ATTACK_CONFIG_PATH=./configs/attack_config2.yaml
CSV_PATH=mapillary_vistas_final_merged.csv
BG_PATH=~/data/mtsd_v2_fully_annotated/test/
IMG_SIZE=1536,2048 # sizes: (1536,2048), (3040,4032)
INTERP=bilinear
OBJ_CLASS=0
# EXP_NAME=none
# EXP_NAME=per-sign_10x10_bottom
# EXP_NAME=per_sign-2_10x20
EXP_NAME=debug

# Test a detector on Detectron2 without attack
# CUDA_VISIBLE_DEVICES=$GPU python -u test_detectron.py \
#     --num-gpus $NUM_GPU --config-file $DETECTRON_CONFIG \
#     --padded-imgsz $IMG_SIZE --name no_patch --tgt-csv-filepath $CSV_PATH \
#     --dataset $DATASET --eval-mode drop --verbose \
#     MODEL.ROI_HEADS.NUM_CLASSES $NUM_CLASSES \
#     OUTPUT_DIR $OUTPUT_PATH \
#     MODEL.WEIGHTS $OUTPUT_PATH/model_best.pth \
#     DATALOADER.NUM_WORKERS 8
# SOLVER.IMS_PER_BATCH 5 \
# --debug --resume --annotated-signs-only
# --dataset mtsd-val-no_color, mapillary-combined-no_color

# Generate mask for adversarial patch
# CUDA_VISIBLE_DEVICES=$GPU python -u gen_mask.py \
#     --dataset $DATASET --patch-name $EXP_NAME --obj-class $OBJ_CLASS --obj-size 256 --save-mask

# Generate adversarial patch (add --synthetic for synthetic attack)
# CUDA_VISIBLE_DEVICES=$GPU python -u gen_patch_yolo.py \
#     --device $GPU --dataset $DATASET --padded-imgsz $IMG_SIZE --name $EXP_NAME \
#     --weights $MODEL_PATH --workers 6 --plot-class-examples $OBJ_CLASS \
#     --tgt-csv-filepath $CSV_PATH --attack-config-path $ATTACK_CONFIG_PATH \
#     --obj-class $OBJ_CLASS --bg-dir $BG_PATH --interp $INTERP --verbose
# --imgsz $YOLO_IMG_SIZE

# Test the generated patch
CUDA_VISIBLE_DEVICES=$GPU python -u test_yolo.py \
    --device $GPU --interp $INTERP --save-exp-metrics --workers 8 \
    --dataset $DATASET --padded-imgsz $IMG_SIZE \
    --tgt-csv-filepath $CSV_PATH --attack-config-path $ATTACK_CONFIG_PATH \
    --name $EXP_NAME --obj-class $OBJ_CLASS --metrics-confidence-threshold $CONF_THRES \
    --annotated-signs-only --plot-class-examples $OBJ_CLASS --batch-size 16 \
    --attack-type debug --transform-mode perspective --verbose

# --data mapillary_no_color.yaml --dataset mapillary-combined-no_color --task test \
#     --tgt-csv-filepath $CSV_PATH
# --weights $MODEL_PATH --exist-ok --workers 8 \
#     --attack-config-path attack_config.yaml --name $PATCH_NAME \
#     --obj-class $OBJ_CLASS --plot-class-examples $OBJ_CLASS --syn-obj-path $SYN_OBJ_PATH \
#     --imgsz $YOLO_IMG_SIZE --padded-imgsz $IMG_SIZE --batch-size 16 \
#     --metrics-confidence-threshold 0.571 \
#     --annotated-signs-only \
#     --attack-type load --adv-patch-path ./runs/val/$PATCH_NAME/$SHAPE/adv_patch.pkl --interp bilinear \
#     --img-txt-path $BG_FILE_PATH --debug
