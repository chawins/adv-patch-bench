# Test attack
CUDA_VISIBLE_DEVICES=0 python val_attack.py \
--img 1280 \
--batch 2 \
--data mtsd.yaml \
--weights /data/chawin/adv-patch-bench/yolov5/runs/train/exp/weights/best.pt \
--exist-ok \
--workers 8