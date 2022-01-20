# Train
CUDA_VISIBLE_DEVICES=0,1 torchrun \
    --standalone --nnodes=1 --max_restarts 0 --nproc_per_node 2 \
    train_yolov5.py \
    --hyp yolov5/data/hyps/hyp.scratch.yaml \
    --img 1280 \
    --batch 40 \
    --data yolov5/data/mtsd.yaml \
    --weights yolov5s.pt \
    --exist-ok \
    --workers 8 \
    --device 0,1 

# CUDA_VISIBLE_DEVICES=0,1 python \
#     train_yolov5.py \
#     --hyp yolov5/data/hyps/hyp.scratch.yaml \
#     --img 1280 \
#     --batch 32 \
#     --data yolov5/data/mtsd.yaml \
#     --weights yolov5s.pt \
#     --exist-ok \
#     --workers 8 \
#     --device 0,1 

# CUDA_VISIBLE_DEVICES=0 python val.py \
# --img 1280 \
# --batch 3 \
# --data mtsd.yaml \
# --weights /home/chawin/adv-patch-bench/yolov5/runs/train/exp/weights/best.pt \
# --exist-ok \
# --workers 8