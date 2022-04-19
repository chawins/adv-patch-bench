# Detector train script
GPU=0,1,2,3
NUM_GPU=4
EXP=faster_rcnn_R_50_FPN_mtsd_orig
# OUTPUT_PATH=~/data/adv-patch-bench/detectron_output/$EXP
OUTPUT_PATH=~/adv-patch-bench/detectron_output/$EXP

# Test a detector on Detectron2
CUDA_VISIBLE_DEVICES=$GPU python -u test_detectron.py \
    --num-gpus $NUM_GPU \
    --config-file ./configs/faster_rcnn_R_50_FPN_3x.yaml \
    OUTPUT_DIR $OUTPUT_PATH \
    MODEL.ROI_HEADS.NUM_CLASSES 401 \
    MODEL.WEIGHTS $OUTPUT_PATH/model_final.pth \
    SOLVER.IMS_PER_BATCH 5 \
    DATALOADER.NUM_WORKERS 32
# MODEL.ROI_HEADS.NUM_CLASSES 15
# MODEL.ROI_HEADS.IOU_THRESHOLDS = [0.5]
# MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.5
# MODEL.WEIGHTS
# --resume --eval-only
# --eval-only MODEL.WEIGHTS /path/to/checkpoint_file
