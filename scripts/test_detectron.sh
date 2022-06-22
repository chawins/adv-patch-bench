# Detector test script
GPU=0
NUM_GPU=1
NUM_WORKERS=6

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
# BG_FILES=synthetic-2_10x20/octagon-915.0/bg_filenames-1.txt
IMG_SIZE=1536,2048 # sizes: (1536,2048), (3040,4032)
INTERP=bilinear
SYN_OBJ_SIZE=256
OBJ_CLASS=10
# EXP_NAME=none
EXP_NAME=real-10x10_bottom
# EXP_NAME=synthetic-2_10x20
# EXP_NAME=debug

# MODEL.ROI_HEADS.NUM_CLASSES 11(12), 15(16), 401  # This should match model not data
# MODEL.ROI_HEADS.IOU_THRESHOLDS = [0.5]
# MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.5

# Test a detector on Detectron2 without attack
# CUDA_VISIBLE_DEVICES=$GPU python -u test_detectron.py \
#     --num-gpus $NUM_GPU --config-file $DETECTRON_CONFIG_PATH --name no_patch \
#     --padded-imgsz $IMG_SIZE --tgt-csv-filepath $CSV_PATH --dataset $DATASET \
#     --attack-config-path $ATTACK_CONFIG_PATH --eval-mode drop --verbose \
#     MODEL.ROI_HEADS.NUM_CLASSES $NUM_CLASSES \
#     OUTPUT_DIR $OUTPUT_PATH \
#     MODEL.WEIGHTS $OUTPUT_PATH/model_best.pth \
#     DATALOADER.NUM_WORKERS $NUM_WORKERS
# SOLVER.IMS_PER_BATCH 5
# --debug --resume --annotated-signs-only
# --dataset mtsd-val-no_color, mapillary-combined-no_color

# Generate mask for adversarial patch
# CUDA_VISIBLE_DEVICES=$GPU python -u gen_mask.py \
#     --dataset $DATASET --patch-name $EXP_NAME --obj-class $OBJ_CLASS --obj-size 256 --save-mask

# Generate adversarial patch (add --synthetic for synthetic attack)
# CUDA_VISIBLE_DEVICES=$GPU python -u gen_patch_detectron.py \
#     --num-gpus $NUM_GPU --config-file $DETECTRON_CONFIG_PATH --interp $INTERP \
#     --dataset $DATASET --padded-imgsz $IMG_SIZE --tgt-csv-filepath $CSV_PATH \
#     --attack-config-path $ATTACK_CONFIG_PATH --obj-class $OBJ_CLASS \
#     --name $EXP_NAME --bg-dir $BG_PATH --transform-mode perspective --verbose \
#     MODEL.ROI_HEADS.NUM_CLASSES $NUM_CLASSES \
#     OUTPUT_DIR $OUTPUT_PATH \
#     MODEL.WEIGHTS $OUTPUT_PATH/model_best.pth \
#     DATALOADER.NUM_WORKERS $NUM_WORKERS

# Test the generated patch
# CUDA_VISIBLE_DEVICES=$GPU python -u test_detectron.py \
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
# CUDA_VISIBLE_DEVICES=$GPU python -u gen_patch_detectron.py \
#     --num-gpus $NUM_GPU --config-file $DETECTRON_CONFIG_PATH --interp $INTERP \
#     --dataset $DATASET --padded-imgsz $IMG_SIZE --tgt-csv-filepath $CSV_PATH \
#     --attack-config-path $ATTACK_CONFIG_PATH --obj-class $OBJ_CLASS \
#     --name $EXP_NAME --bg-dir $BG_PATH --transform-mode perspective --verbose --synthetic \
#     MODEL.ROI_HEADS.NUM_CLASSES $NUM_CLASSES \
#     OUTPUT_DIR $OUTPUT_PATH \
#     MODEL.WEIGHTS $OUTPUT_PATH/model_best.pth \
#     DATALOADER.NUM_WORKERS $NUM_WORKERS

# Test the generated patch
# CUDA_VISIBLE_DEVICES=$GPU python -u test_detectron.py \
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

# =========================================================================== #

# EXP_NAME=real-10x10_bottom

# CUDA_VISIBLE_DEVICES=$GPU python -u gen_patch_detectron.py \
#     --num-gpus $NUM_GPU --config-file $DETECTRON_CONFIG_PATH --interp $INTERP \
#     --dataset $DATASET --padded-imgsz $IMG_SIZE --tgt-csv-filepath $CSV_PATH \
#     --attack-config-path $ATTACK_CONFIG_PATH --obj-class $OBJ_CLASS \
#     --name $EXP_NAME --bg-dir $BG_PATH --transform-mode perspective --verbose \
#     --save-images \
#     MODEL.ROI_HEADS.NUM_CLASSES $NUM_CLASSES \
#     OUTPUT_DIR $OUTPUT_PATH \
#     MODEL.WEIGHTS $OUTPUT_PATH/model_best.pth \
#     DATALOADER.NUM_WORKERS $NUM_WORKERS
# # --img-txt-path $BG_FILES

# CUDA_VISIBLE_DEVICES=$GPU python -u test_detectron.py \
#     --num-gpus $NUM_GPU --config-file $DETECTRON_CONFIG_PATH --interp $INTERP \
#     --dataset $DATASET --padded-imgsz $IMG_SIZE --eval-mode drop \
#     --tgt-csv-filepath $CSV_PATH --attack-config-path $ATTACK_CONFIG_PATH \
#     --name $EXP_NAME --obj-class $OBJ_CLASS --conf-thres $CONF_THRES \
#     --annotated-signs-only --transform-mode perspective --attack-type load \
#     --verbose --debug \
#     MODEL.ROI_HEADS.NUM_CLASSES $NUM_CLASSES \
#     OUTPUT_DIR $OUTPUT_PATH \
#     MODEL.WEIGHTS $OUTPUT_PATH/model_best.pth \
#     DATALOADER.NUM_WORKERS $NUM_WORKERS
# # --img-txt-path $BG_FILES

EXP_NAME=synthetic-2_10x20-adam-lr1e-1
# cp -r ./detectron_output/synthetic-10x10_bottom/* ./detectron_output/$EXP_NAME

CUDA_VISIBLE_DEVICES=$GPU python -u gen_patch_detectron.py \
    --num-gpus $NUM_GPU --config-file $DETECTRON_CONFIG_PATH --interp $INTERP \
    --dataset $DATASET --padded-imgsz $IMG_SIZE --tgt-csv-filepath $CSV_PATH \
    --attack-config-path $ATTACK_CONFIG_PATH --obj-class $OBJ_CLASS \
    --name $EXP_NAME --bg-dir $BG_PATH --transform-mode perspective --verbose \
    --obj-size $SYN_OBJ_SIZE --save-images --synthetic \
    MODEL.ROI_HEADS.NUM_CLASSES $NUM_CLASSES \
    OUTPUT_DIR $OUTPUT_PATH \
    MODEL.WEIGHTS $OUTPUT_PATH/model_best.pth \
    DATALOADER.NUM_WORKERS $NUM_WORKERS
# --img-txt-path $BG_FILES --synthetic --obj-size 64 \

CUDA_VISIBLE_DEVICES=$GPU python -u test_detectron.py \
    --num-gpus $NUM_GPU --config-file $DETECTRON_CONFIG_PATH --interp $INTERP \
    --dataset $DATASET --padded-imgsz $IMG_SIZE --eval-mode drop \
    --tgt-csv-filepath $CSV_PATH --attack-config-path $ATTACK_CONFIG_PATH \
    --name $EXP_NAME --obj-class $OBJ_CLASS --conf-thres $CONF_THRES \
    --annotated-signs-only --transform-mode perspective --attack-type load \
    --img-txt-path $BG_FILES --synthetic \
    MODEL.ROI_HEADS.NUM_CLASSES $NUM_CLASSES \
    OUTPUT_DIR $OUTPUT_PATH \
    MODEL.WEIGHTS $OUTPUT_PATH/model_best.pth \
    DATALOADER.NUM_WORKERS $NUM_WORKERS

CUDA_VISIBLE_DEVICES=$GPU python -u test_detectron.py \
    --num-gpus $NUM_GPU --config-file $DETECTRON_CONFIG_PATH --interp $INTERP \
    --dataset $DATASET --padded-imgsz $IMG_SIZE --eval-mode drop \
    --tgt-csv-filepath $CSV_PATH --attack-config-path $ATTACK_CONFIG_PATH \
    --name $EXP_NAME --obj-class $OBJ_CLASS --conf-thres $CONF_THRES \
    --annotated-signs-only --transform-mode perspective --attack-type load \
    --img-txt-path $BG_FILES \
    MODEL.ROI_HEADS.NUM_CLASSES $NUM_CLASSES \
    OUTPUT_DIR $OUTPUT_PATH \
    MODEL.WEIGHTS $OUTPUT_PATH/model_best.pth \
    DATALOADER.NUM_WORKERS $NUM_WORKERS

EXP_NAME=synthetic-2_10x20-adam-lr1e-2
ATTACK_CONFIG_PATH=./configs/attack_config3.yaml
# cp -r ./detectron_output/synthetic-10x10_bottom/* ./detectron_output/$EXP_NAME

CUDA_VISIBLE_DEVICES=$GPU python -u gen_patch_detectron.py \
    --num-gpus $NUM_GPU --config-file $DETECTRON_CONFIG_PATH --interp $INTERP \
    --dataset $DATASET --padded-imgsz $IMG_SIZE --tgt-csv-filepath $CSV_PATH \
    --attack-config-path $ATTACK_CONFIG_PATH --obj-class $OBJ_CLASS \
    --name $EXP_NAME --bg-dir $BG_PATH --transform-mode perspective --verbose \
    --obj-size $SYN_OBJ_SIZE --save-images --img-txt-path $BG_FILES --synthetic \
    MODEL.ROI_HEADS.NUM_CLASSES $NUM_CLASSES \
    OUTPUT_DIR $OUTPUT_PATH \
    MODEL.WEIGHTS $OUTPUT_PATH/model_best.pth \
    DATALOADER.NUM_WORKERS $NUM_WORKERS
# --img-txt-path $BG_FILES --synthetic --obj-size 64 \

CUDA_VISIBLE_DEVICES=$GPU python -u test_detectron.py \
    --num-gpus $NUM_GPU --config-file $DETECTRON_CONFIG_PATH --interp $INTERP \
    --dataset $DATASET --padded-imgsz $IMG_SIZE --eval-mode drop \
    --tgt-csv-filepath $CSV_PATH --attack-config-path $ATTACK_CONFIG_PATH \
    --name $EXP_NAME --obj-class $OBJ_CLASS --conf-thres $CONF_THRES \
    --annotated-signs-only --transform-mode perspective --attack-type load \
    --img-txt-path $BG_FILES --synthetic \
    MODEL.ROI_HEADS.NUM_CLASSES $NUM_CLASSES \
    OUTPUT_DIR $OUTPUT_PATH \
    MODEL.WEIGHTS $OUTPUT_PATH/model_best.pth \
    DATALOADER.NUM_WORKERS $NUM_WORKERS

CUDA_VISIBLE_DEVICES=$GPU python -u test_detectron.py \
    --num-gpus $NUM_GPU --config-file $DETECTRON_CONFIG_PATH --interp $INTERP \
    --dataset $DATASET --padded-imgsz $IMG_SIZE --eval-mode drop \
    --tgt-csv-filepath $CSV_PATH --attack-config-path $ATTACK_CONFIG_PATH \
    --name $EXP_NAME --obj-class $OBJ_CLASS --conf-thres $CONF_THRES \
    --annotated-signs-only --transform-mode perspective --attack-type load \
    --img-txt-path $BG_FILES \
    MODEL.ROI_HEADS.NUM_CLASSES $NUM_CLASSES \
    OUTPUT_DIR $OUTPUT_PATH \
    MODEL.WEIGHTS $OUTPUT_PATH/model_best.pth \
    DATALOADER.NUM_WORKERS $NUM_WORKERS

EXP_NAME=synthetic-2_10x20-adam-lr1e-3
ATTACK_CONFIG_PATH=./configs/attack_config4.yaml
# cp -r ./detectron_output/synthetic-10x10_bottom/* ./detectron_output/$EXP_NAME

CUDA_VISIBLE_DEVICES=$GPU python -u gen_patch_detectron.py \
    --num-gpus $NUM_GPU --config-file $DETECTRON_CONFIG_PATH --interp $INTERP \
    --dataset $DATASET --padded-imgsz $IMG_SIZE --tgt-csv-filepath $CSV_PATH \
    --attack-config-path $ATTACK_CONFIG_PATH --obj-class $OBJ_CLASS \
    --name $EXP_NAME --bg-dir $BG_PATH --transform-mode perspective --verbose \
    --obj-size $SYN_OBJ_SIZE --save-images --img-txt-path $BG_FILES --synthetic \
    MODEL.ROI_HEADS.NUM_CLASSES $NUM_CLASSES \
    OUTPUT_DIR $OUTPUT_PATH \
    MODEL.WEIGHTS $OUTPUT_PATH/model_best.pth \
    DATALOADER.NUM_WORKERS $NUM_WORKERS
# --img-txt-path $BG_FILES --synthetic --obj-size 64 \

CUDA_VISIBLE_DEVICES=$GPU python -u test_detectron.py \
    --num-gpus $NUM_GPU --config-file $DETECTRON_CONFIG_PATH --interp $INTERP \
    --dataset $DATASET --padded-imgsz $IMG_SIZE --eval-mode drop \
    --tgt-csv-filepath $CSV_PATH --attack-config-path $ATTACK_CONFIG_PATH \
    --name $EXP_NAME --obj-class $OBJ_CLASS --conf-thres $CONF_THRES \
    --annotated-signs-only --transform-mode perspective --attack-type load \
    --img-txt-path $BG_FILES --synthetic \
    MODEL.ROI_HEADS.NUM_CLASSES $NUM_CLASSES \
    OUTPUT_DIR $OUTPUT_PATH \
    MODEL.WEIGHTS $OUTPUT_PATH/model_best.pth \
    DATALOADER.NUM_WORKERS $NUM_WORKERS

CUDA_VISIBLE_DEVICES=$GPU python -u test_detectron.py \
    --num-gpus $NUM_GPU --config-file $DETECTRON_CONFIG_PATH --interp $INTERP \
    --dataset $DATASET --padded-imgsz $IMG_SIZE --eval-mode drop \
    --tgt-csv-filepath $CSV_PATH --attack-config-path $ATTACK_CONFIG_PATH \
    --name $EXP_NAME --obj-class $OBJ_CLASS --conf-thres $CONF_THRES \
    --annotated-signs-only --transform-mode perspective --attack-type load \
    --img-txt-path $BG_FILES \
    MODEL.ROI_HEADS.NUM_CLASSES $NUM_CLASSES \
    OUTPUT_DIR $OUTPUT_PATH \
    MODEL.WEIGHTS $OUTPUT_PATH/model_best.pth \
    DATALOADER.NUM_WORKERS $NUM_WORKERS

EXP_NAME=synthetic-2_10x20-pgd-lr1e-2
ATTACK_CONFIG_PATH=./configs/attack_config5.yaml
# cp -r ./detectron_output/synthetic-10x10_bottom/* ./detectron_output/$EXP_NAME

CUDA_VISIBLE_DEVICES=$GPU python -u gen_patch_detectron.py \
    --num-gpus $NUM_GPU --config-file $DETECTRON_CONFIG_PATH --interp $INTERP \
    --dataset $DATASET --padded-imgsz $IMG_SIZE --tgt-csv-filepath $CSV_PATH \
    --attack-config-path $ATTACK_CONFIG_PATH --obj-class $OBJ_CLASS \
    --name $EXP_NAME --bg-dir $BG_PATH --transform-mode perspective --verbose \
    --obj-size $SYN_OBJ_SIZE --save-images --img-txt-path $BG_FILES --synthetic \
    MODEL.ROI_HEADS.NUM_CLASSES $NUM_CLASSES \
    OUTPUT_DIR $OUTPUT_PATH \
    MODEL.WEIGHTS $OUTPUT_PATH/model_best.pth \
    DATALOADER.NUM_WORKERS $NUM_WORKERS
# --img-txt-path $BG_FILES --synthetic --obj-size 64 \

CUDA_VISIBLE_DEVICES=$GPU python -u test_detectron.py \
    --num-gpus $NUM_GPU --config-file $DETECTRON_CONFIG_PATH --interp $INTERP \
    --dataset $DATASET --padded-imgsz $IMG_SIZE --eval-mode drop \
    --tgt-csv-filepath $CSV_PATH --attack-config-path $ATTACK_CONFIG_PATH \
    --name $EXP_NAME --obj-class $OBJ_CLASS --conf-thres $CONF_THRES \
    --annotated-signs-only --transform-mode perspective --attack-type load \
    --img-txt-path $BG_FILES --synthetic \
    MODEL.ROI_HEADS.NUM_CLASSES $NUM_CLASSES \
    OUTPUT_DIR $OUTPUT_PATH \
    MODEL.WEIGHTS $OUTPUT_PATH/model_best.pth \
    DATALOADER.NUM_WORKERS $NUM_WORKERS

CUDA_VISIBLE_DEVICES=$GPU python -u test_detectron.py \
    --num-gpus $NUM_GPU --config-file $DETECTRON_CONFIG_PATH --interp $INTERP \
    --dataset $DATASET --padded-imgsz $IMG_SIZE --eval-mode drop \
    --tgt-csv-filepath $CSV_PATH --attack-config-path $ATTACK_CONFIG_PATH \
    --name $EXP_NAME --obj-class $OBJ_CLASS --conf-thres $CONF_THRES \
    --annotated-signs-only --transform-mode perspective --attack-type load \
    --img-txt-path $BG_FILES \
    MODEL.ROI_HEADS.NUM_CLASSES $NUM_CLASSES \
    OUTPUT_DIR $OUTPUT_PATH \
    MODEL.WEIGHTS $OUTPUT_PATH/model_best.pth \
    DATALOADER.NUM_WORKERS $NUM_WORKERS
