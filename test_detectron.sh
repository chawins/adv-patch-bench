# Detector train script
GPU=1
NUM_GPU=1
EXP=faster_rcnn_R_50_FPN_mtsd_no_color_1
# EXP=faster_rcnn_R_50_FPN_mtsd_color_1
OUTPUT_PATH=~/adv-patch-bench/detectron_output/$EXP

# Test a detector on Detectron2
CUDA_VISIBLE_DEVICES=$GPU python -u test_detectron.py \
    --num-gpus $NUM_GPU \
    --config-file ./configs/faster_rcnn_R_50_FPN_3x.yaml \
    --dataset mtsd_no_color --eval-mode drop \
    --debug \
    MODEL.ROI_HEADS.NUM_CLASSES 11 \
    OUTPUT_DIR $OUTPUT_PATH \
    MODEL.WEIGHTS $OUTPUT_PATH/model_final.pth \
    DATALOADER.NUM_WORKERS 8
# --debug \
# --single-image \
# --data-no-other \
# SOLVER.IMS_PER_BATCH 5 \
# MODEL.ROI_HEADS.NUM_CLASSES 11(12), 15(16), 401  # This should match model not data
# MODEL.ROI_HEADS.IOU_THRESHOLDS = [0.5]
# MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.5
# MODEL.WEIGHTS
# --resume --eval-only
# --eval-only MODEL.WEIGHTS /path/to/checkpoint_file
