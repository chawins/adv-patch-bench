# Detector train script
GPU=0
NUM_GPU=1
EXP=faster_rcnn_R_50_FPN_mtsd_no_color_2
# EXP=faster_rcnn_R_50_FPN_mtsd_color_1
OUTPUT_PATH=~/adv-patch-bench/detectron_output/$EXP
CSV_PATH=mapillary_vistas_final_merged.csv
SYN_OBJ_PATH=attack_assets/octagon-915.0.png
OBJ_CLASS=-1
IMG_SIZE=1536,2048 # sizes: (1536,2048), (3040,4032)

# EXP_NAME=per-sign_10x10_bottom
EXP_NAME=none

# MODEL.ROI_HEADS.NUM_CLASSES 11(12), 15(16), 401  # This should match model not data
# MODEL.ROI_HEADS.IOU_THRESHOLDS = [0.5]
# MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.5

# Test a detector on Detectron2 without attack
# CUDA_VISIBLE_DEVICES=$GPU python -u test_detectron.py \
#     --num-gpus $NUM_GPU --config-file ./configs/faster_rcnn_R_50_FPN_3x.yaml \
#     --padded-imgsz $IMG_SIZE --name no_patch --tgt-csv-filepath $CSV_PATH \
#     --dataset mtsd-val-no_color --eval-mode drop --verbose \
#     MODEL.ROI_HEADS.NUM_CLASSES 12 \
#     OUTPUT_DIR $OUTPUT_PATH \
#     MODEL.WEIGHTS $OUTPUT_PATH/model_final.pth \
#     DATALOADER.NUM_WORKERS 8
# SOLVER.IMS_PER_BATCH 5 \
# --debug --resume --annotated-signs-only
# --dataset mtsd-val-no_color, mapillary-combined-no_color

# Generate mask for adversarial patch
# CUDA_VISIBLE_DEVICES=$GPU python -u gen_mask.py \
#     --syn-obj-path $SYN_OBJ_PATH --dataset mapillary-combined-no_color \
#     --patch-name $EXP_NAME --obj-class $OBJ_CLASS --obj-size 256 --save-mask

# Generate adversarial patch --synthetic
# CUDA_VISIBLE_DEVICES=$GPU python -u gen_patch_detectron.py \
#     --num-gpus $NUM_GPU --config-file ./configs/faster_rcnn_R_50_FPN_3x.yaml \
#     --dataset mapillary-combined-no_color --padded-imgsz $IMG_SIZE \
#     --bg-dir ~/data/mtsd_v2_fully_annotated/test/ --num-bg 1 \
#     --tgt-csv-filepath $CSV_PATH --attack-config-path attack_config2.yaml \
#     --name $EXP_NAME \
#     --obj-class $OBJ_CLASS --syn-obj-path $SYN_OBJ_PATH --verbose --debug \
#     MODEL.ROI_HEADS.NUM_CLASSES 12 \
#     OUTPUT_DIR $OUTPUT_PATH \
#     MODEL.WEIGHTS $OUTPUT_PATH/model_best.pth \
#     DATALOADER.NUM_WORKERS 8

# Test the generated patch
CUDA_VISIBLE_DEVICES=$GPU python -u test_detectron.py \
    --num-gpus $NUM_GPU --config-file ./configs/faster_rcnn_R_50_FPN_3x.yaml \
    --dataset mapillary-combined-no_color --padded-imgsz $IMG_SIZE --eval-mode drop \
    --tgt-csv-filepath $CSV_PATH --attack-config-path attack_config.yaml \
    --name $EXP_NAME --obj-class $OBJ_CLASS --syn-obj-path $SYN_OBJ_PATH \
    --attack-type none --verbose --annotated-signs-only --interp bilinear --debug \
    MODEL.ROI_HEADS.NUM_CLASSES 12 \
    OUTPUT_DIR $OUTPUT_PATH \
    MODEL.WEIGHTS $OUTPUT_PATH/model_best.pth \
    DATALOADER.NUM_WORKERS 6
# --adv-patch-path ./runs/val/exp$EXP/$EXP_NAME.pkl --name $EXP_NAME \
# --compute-metrics --min-area 600
# --interp bilinear
