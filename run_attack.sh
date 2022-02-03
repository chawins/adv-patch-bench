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
# --apply_patch

# CUDA_VISIBLE_DEVICES=0 python val_attack_synthetic.py \
# --img 1280 \
# --batch-size 16 \
# --data mapillary_vistas.yaml \
# --weights /home/chawins/adv-patch-bench/yolov5/runs/train/exp2/weights/best.pt \
# --exist-ok \
# --workers 8 \
# --task train \
# --save_exp_metrics \
# --apply_patch

CUDA_VISIBLE_DEVICES=0 python val_attack_synthetic.py \
--img 1280 \
--batch-size 16 \
--data mapillary_vistas.yaml \
--weights /home/chawins/adv-patch-bench/yolov5/runs/train/exp2/weights/best.pt \
--exist-ok \
--workers 8 \
--task train \
--save_exp_metrics \
--num_bg 16 \
--apply_patch \
--plot_octagons

# CUDA_VISIBLE_DEVICES=0 python val_attack_synthetic.py \
# --img 1280 \
# --batch-size 16 \
# --data mapillary_vistas.yaml \
# --weights /home/chawins/adv-patch-bench/yolov5/runs/train/exp2/weights/best.pt \
# --exist-ok \
# --workers 8 \
# --task train \
# --save_exp_metrics \
# --synthetic \
# --num_bg 16 \
# --apply_patch \


# CUDA_VISIBLE_DEVICES=0 python val_attack_synthetic.py \
# --img 1280 \
# --batch-size 16 \
# --data mapillary_vistas.yaml \
# --weights /home/chawins/adv-patch-bench/yolov5/runs/train/exp2/weights/best.pt \
# --exist-ok \
# --workers 8 \
# --task train \
# --save_exp_metrics \
# --synthetic \
# --apply_patch \
# --num_bg 32

# CUDA_VISIBLE_DEVICES=0 python val_attack_synthetic.py \
# --img 1280 \
# --batch-size 16 \
# --data mapillary_vistas.yaml \
# --weights /home/chawins/adv-patch-bench/yolov5/runs/train/exp2/weights/best.pt \
# --exist-ok \
# --workers 8 \
# --task train \
# --save_exp_metrics \
# --synthetic \
# --apply_patch \
# --num_bg 100

# CUDA_VISIBLE_DEVICES=0 python val_attack_synthetic.py \
# --img 1280 \
# --batch-size 16 \
# --data mapillary_vistas.yaml \
# --weights /home/chawins/adv-patch-bench/yolov5/runs/train/exp2/weights/best.pt \
# --exist-ok \
# --workers 8 \
# --task train \
# --save_exp_metrics \
# --synthetic \
# --apply_patch \
# --num_bg 500

# CUDA_VISIBLE_DEVICES=0 python val_attack_synthetic.py \
# --img 1280 \
# --batch-size 16 \
# --data mapillary_vistas.yaml \
# --weights /home/chawins/adv-patch-bench/yolov5/runs/train/exp2/weights/best.pt \
# --exist-ok \
# --workers 8 \
# --task train \
# --save_exp_metrics \
# --synthetic \
# --apply_patch \
# --num_bg 1000


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
# --save_exp_metrics \
# --apply_patch 

# CUDA_VISIBLE_DEVICES=0 python val_attack_synthetic.py \
# --img 1280 \
# --batch-size 16 \
# --data mtsd.yaml \
# --weights /data/chawin/adv-patch-bench/yolov5/runs/train/exp2/weights/best.pt \
# --exist-ok \
# --workers 8 \
# --task train \
# --save_exp_metrics \
# --apply_patch \
# --random_patch


# CUDA_VISIBLE_DEVICES=0 python val_attack_synthetic.py \
# --img 1280 \
# --batch-size 16 \
# --data mapillary_vistas.yaml \
# --weights /home/chawins/adv-patch-bench/yolov5/runs/train/exp2/weights/best.pt \
# --exist-ok \
# --workers 8 \
# --task train \
# --save_exp_metrics 

# CUDA_VISIBLE_DEVICES=0 python val_attack_synthetic.py \
# --img 1280 \
# --batch-size 16 \
# --data mapillary_vistas.yaml \
# --weights /data/chawin/adv-patch-bench/yolov5/runs/train/exp2/weights/best.pt \
# --exist-ok \
# --workers 8 \
# --task train \
# --save_exp_metrics 

# CUDA_VISIBLE_DEVICES=0 python val_attack_synthetic.py \
# --img 1280 \
# --batch-size 16 \
# --data mapillary_vistas.yaml \
# --weights /data/chawin/adv-patch-bench/yolov5/runs/train/exp2/weights/best.pt \
# --exist-ok \
# --workers 8 \
# --task train \
# --save_exp_metrics \
# --apply_patch 

# CUDA_VISIBLE_DEVICES=0 python val_attack_synthetic.py \
# --img 1280 \
# --batch-size 16 \
# --data mapillary_vistas.yaml \
# --weights /data/chawin/adv-patch-bench/yolov5/runs/train/exp2/weights/best.pt \
# --exist-ok \
# --workers 8 \
# --task train \
# --save_exp_metrics \
# --apply_patch \
# --random_patch

# CUDA_VISIBLE_DEVICES=0 python val_attack_synthetic.py \
# --img 1280 \
# --batch-size 16 \
# --data mapillary_vistas.yaml \
# --weights /data/chawin/adv-patch-bench/yolov5/runs/train/exp2/weights/best.pt \
# --exist-ok \
# --workers 8 \
# --task train \
# --apply_patch \
# --random_patch \
# --save_exp_metrics

# --weights /data/chawin/adv-patch-bench/yolov5/runs/train/exp/weights/best.pt \


# CUDA_VISIBLE_DEVICES=0 python val.py \
# --img 1280 \
# --batch 3 \
# --data mtsd.yaml \
# --weights /data/chawin/adv-patch-bench/yolov5/runs/train/exp/weights/best.pt \
# --exist-ok \
# --workers 8