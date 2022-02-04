# Test attack
# CUDA_VISIBLE_DEVICES=0 python val_attack.py \
# --img 1280 \
# --batch 2 \
# --data mtsd.yaml \
# --weights /data/chawin/adv-patch-bench/yolov5/runs/train/exp/weights/best.pt \
# --exist-ok \
# --workers 8

# Test attack
# CUDA_VISIBLE_DEVICES=1 python test_rp2_attack.py \
# CUDA_VISIBLE_DEVICES=0 python val_attack.py \
# --img 1280 \
# --batch 2 \
# --data mapillary_vistas.yaml \
# --weights /home/chawins/adv-patch-bench/yolov5/runs/train/exp2/weights/best.pt \
# --exist-ok \
# --workers 8 \
# --task train
# --apply-patch

# CUDA_VISIBLE_DEVICES=0 python val_attack_synthetic.py \
# --img 1280 \
# --batch-size 16 \
# --data mapillary_vistas.yaml \
# --weights /home/chawins/adv-patch-bench/yolov5/runs/train/exp2/weights/best.pt \
# --exist-ok \
# --workers 8 \
# --task train \
# --save-exp-metrics \
# --apply-patch

CUDA_VISIBLE_DEVICES=0 python val_attack_synthetic.py \
    --img 1280 \
    --batch-size 8 \
    --data mapillary_vistas.yaml \
    --weights /home/chawins/adv-patch-bench/yolov5/runs/train/exp2/weights/best.pt \
    --exist-ok \
    --workers 8 \
    --task train \
    --save-exp-metrics \
    --num-bg 16 \
    --plot-octagons \
    --apply-patch \
    --load-patch ./adv_patch.pkl

# CUDA_VISIBLE_DEVICES=0 python val_attack_synthetic.py \
# --img 1280 \
# --batch-size 16 \
# --data mapillary_vistas.yaml \
# --weights /home/chawins/adv-patch-bench/yolov5/runs/train/exp2/weights/best.pt \
# --exist-ok \
# --workers 8 \
# --task train \
# --save-exp-metrics \
# --synthetic \
# --num-bg 16 \
# --apply-patch \

# CUDA_VISIBLE_DEVICES=0 python val_attack_synthetic.py \
# --img 1280 \
# --batch-size 16 \
# --data mapillary_vistas.yaml \
# --weights /home/chawins/adv-patch-bench/yolov5/runs/train/exp2/weights/best.pt \
# --exist-ok \
# --workers 8 \
# --task train \
# --save-exp-metrics \
# --synthetic \
# --apply-patch \
# --num-bg 32

# CUDA_VISIBLE_DEVICES=0 python val_attack_synthetic.py \
# --img 1280 \
# --batch-size 16 \
# --data mapillary_vistas.yaml \
# --weights /home/chawins/adv-patch-bench/yolov5/runs/train/exp2/weights/best.pt \
# --exist-ok \
# --workers 8 \
# --task train \
# --save-exp-metrics \
# --synthetic \
# --apply-patch \
# --num-bg 100

# CUDA_VISIBLE_DEVICES=0 python val_attack_synthetic.py \
# --img 1280 \
# --batch-size 16 \
# --data mapillary_vistas.yaml \
# --weights /home/chawins/adv-patch-bench/yolov5/runs/train/exp2/weights/best.pt \
# --exist-ok \
# --workers 8 \
# --task train \
# --save-exp-metrics \
# --synthetic \
# --apply-patch \
# --num-bg 500

# CUDA_VISIBLE_DEVICES=0 python val_attack_synthetic.py \
# --img 1280 \
# --batch-size 16 \
# --data mapillary_vistas.yaml \
# --weights /home/chawins/adv-patch-bench/yolov5/runs/train/exp2/weights/best.pt \
# --exist-ok \
# --workers 8 \
# --task train \
# --save-exp-metrics \
# --synthetic \
# --apply-patch \
# --num-bg 1000

# \
# --plot_single_images

# CUDA_VISIBLE_DEVICES=0 python val_attack_synthetic.py \
# --img 1280 \
# --batch-size 16 \
# --data mtsd.yaml \
# --weights /data/chawin/adv-patch-bench/yolov5/runs/train/exp2/weights/best.pt \
# --exist-ok \
# --workers 8 \
# --task train \
# --save-exp-metrics \
# --apply-patch

# CUDA_VISIBLE_DEVICES=0 python val_attack_synthetic.py \
# --img 1280 \
# --batch-size 16 \
# --data mtsd.yaml \
# --weights /data/chawin/adv-patch-bench/yolov5/runs/train/exp2/weights/best.pt \
# --exist-ok \
# --workers 8 \
# --task train \
# --save-exp-metrics \
# --apply-patch \
# --random_patch

# CUDA_VISIBLE_DEVICES=0 python val_attack_synthetic.py \
# --img 1280 \
# --batch-size 16 \
# --data mapillary_vistas.yaml \
# --weights /home/chawins/adv-patch-bench/yolov5/runs/train/exp2/weights/best.pt \
# --exist-ok \
# --workers 8 \
# --task train \
# --save-exp-metrics

# CUDA_VISIBLE_DEVICES=0 python val_attack_synthetic.py \
# --img 1280 \
# --batch-size 16 \
# --data mapillary_vistas.yaml \
# --weights /data/chawin/adv-patch-bench/yolov5/runs/train/exp2/weights/best.pt \
# --exist-ok \
# --workers 8 \
# --task train \
# --save-exp-metrics

# CUDA_VISIBLE_DEVICES=0 python val_attack_synthetic.py \
# --img 1280 \
# --batch-size 16 \
# --data mapillary_vistas.yaml \
# --weights /data/chawin/adv-patch-bench/yolov5/runs/train/exp2/weights/best.pt \
# --exist-ok \
# --workers 8 \
# --task train \
# --save-exp-metrics \
# --apply-patch

# CUDA_VISIBLE_DEVICES=0 python val_attack_synthetic.py \
# --img 1280 \
# --batch-size 16 \
# --data mapillary_vistas.yaml \
# --weights /data/chawin/adv-patch-bench/yolov5/runs/train/exp2/weights/best.pt \
# --exist-ok \
# --workers 8 \
# --task train \
# --save-exp-metrics \
# --apply-patch \
# --random_patch

# CUDA_VISIBLE_DEVICES=0 python val_attack_synthetic.py \
# --img 1280 \
# --batch-size 16 \
# --data mapillary_vistas.yaml \
# --weights /data/chawin/adv-patch-bench/yolov5/runs/train/exp2/weights/best.pt \
# --exist-ok \
# --workers 8 \
# --task train \
# --apply-patch \
# --random_patch \
# --save-exp-metrics

# --weights /data/chawin/adv-patch-bench/yolov5/runs/train/exp/weights/best.pt \

# CUDA_VISIBLE_DEVICES=0 python val.py \
# --img 1280 \
# --batch 3 \
# --data mtsd.yaml \
# --weights /data/chawin/adv-patch-bench/yolov5/runs/train/exp/weights/best.pt \
# --exist-ok \
# --workers 8
