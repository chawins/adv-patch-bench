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

# CUDA_VISIBLE_DEVICES=1 python -u generate_adv_patch.py \
#     --seed 0 \
#     --data mapillary_vistas.yaml \
#     --weights /data/shared/adv-patch-bench/yolov5/runs/train/exp2/weights/best.pt \
#     --patch-name stop_sign_transform \
#     --imgsz 1280 \
#     --padded_imgsz 736,1312 \
#     --obj-class 14 \
#     --obj-size 128 \
#     --obj-path attack_assets/octagon-915.0.png \
#     --num-bg 50 \
#     --bg-dir /data/shared/mtsd_v2_fully_annotated/train \
#     --csv-path mapillary_vistas_final_merged.csv \
#     --save-images \
#     --generate-patch transform






# torch.Size([1, 608, 608])
# torch.Size([1, 736, 1312])

# CUDA_VISIBLE_DEVICES=0 python -u generate_adv_patch.py \
#     --seed 0 \
#     --data mapillary_vistas.yaml \
#     --weights /data/shared/adv-patch-bench/yolov5/runs/train/exp2/weights/best.pt \
#     --patch-name stop_sign_synthetic_generated_v2 \
#     --imgsz 1280 \
#     --obj-class 14 \
#     --obj-size 128 \
#     --obj-path attack_assets/octagon-915.0.png \
#     --num-bg 50 \
#     --bg-dir /data/shared/mtsd_v2_fully_annotated/train \
#     --save-images \
#     --generate-patch synthetic

CUDA_VISIBLE_DEVICES=1 python -u val_attack_synthetic.py \
    --imgsz 1280 \
    --padded_imgsz 736,1312 \
    --batch-size 8 \
    --data mapillary_vistas.yaml \
    --weights /data/shared/adv-patch-bench/yolov5/runs/train/exp2/weights/best.pt \
    --exist-ok \
    --workers 8 \
    --task train \
    --save-exp-metrics \
    --plot-octagons \
    --obj-size 128 \
    --metrics-confidence-threshold 0.359 \
    --apply-patch \
    --load-patch ./runs/val/exp13/stop_sign_synthetic_generated_v2.pkl \
    --synthetic \
    --patch-loc 87 47
    







# CUDA_VISIBLE_DEVICES=1 python -u val_attack_synthetic.py \
#     --imgsz 1280 \
#     --padded_imgsz 736,1312 \
#     --batch-size 8 \
#     --data mapillary_vistas.yaml \
#     --weights /data/shared/adv-patch-bench/yolov5/runs/train/exp2/weights/best.pt \
#     --exist-ok \
#     --workers 8 \
#     --task train \
#     --save-exp-metrics \
#     --plot-octagons \
#     --apply-patch \
#     --load-patch ./runs/val/exp/stop_sign_transform.pkl \
#     --obj-size 128 
    # --img-txt-path ./runs/val/exp14/bg_filenames.txt 
    # --run-only-img-txt

# CUDA_VISIBLE_DEVICES=0 python -u val_attack_synthetic.py \
#     --img 1280 \
#     --batch-size 8 \
#     --data mapillary_vistas.yaml \
#     --weights /data/shared/adv-patch-bench/yolov5/runs/train/exp2/weights/best.pt \
#     --exist-ok \
#     --workers 8 \
#     --task train \
#     --save-exp-metrics \
#     --plot-octagons \
#     --apply-patch \
#     --load-patch ./runs/val/exp5/stop_sign_synthetic_generated_v2.pkl \
#     --obj-size 128 \
#     --patch-loc 29,29 \
#     --patch-size-mm 254.17 

# CUDA_VISIBLE_DEVICES=0 python -u val_attack_synthetic.py \
#     --img 1280 \
#     --batch-size 8 \
#     --data mapillary_vistas.yaml \
#     --weights /data/shared/adv-patch-bench/yolov5/runs/train/exp2/weights/best.pt \
#     --exist-ok \
#     --workers 8 \
#     --task train \
#     --save-exp-metrics \
#     --plot-octagons \
#     --obj-size 128 \
#     --patch-loc 29,29 \
#     --patch-size-mm 254.17 
    
# CUDA_VISIBLE_DEVICES=0 python val_attack_synthetic.py \
#     --img 1280 \
#     --batch-size 8 \
#     --data mapillary_vistas.yaml \
#     --weights /data/shared/adv-patch-bench/yolov5/runs/train/exp2/weights/best.pt \
#     --exist-ok \
#     --workers 8 \
#     --task train \
#     --save-exp-metrics \
#     --num-bg 16 \
#     --plot-octagons \
#     --apply-patch \
#     --load-patch ./adv_patch.pkl

# CUDA_VISIBLE_DEVICES=0 python val_attack_synthetic.py \
#     --img 1280 \
#     --batch-size 8 \
#     --data mapillary_vistas.yaml \
#     --weights /data/shared/adv-patch-bench/yolov5/runs/train/exp2/weights/best.pt \
#     --exist-ok \
#     --workers 8 \
#     --task train \
#     --save-exp-metrics \
#     --num-bg 16 \
#     --plot-octagons

# CUDA_VISIBLE_DEVICES=0 python val_attack_synthetic.py \
#     --img 1280 \
#     --batch-size 8 \
#     --data mapillary_vistas.yaml \
#     --weights /home/chawins/adv-patch-bench/yolov5/runs/train/exp2/weights/best.pt \
#     --exist-ok \
#     --workers 8 \
#     --task train \
#     --save-exp-metrics \
#     --num-bg 16 \
#     --plot-octagons \
#     --apply-patch \
#     --load-patch ./adv_patch.pkl \
#     --synthetic

# CUDA_VISIBLE_DEVICES=0 python val_attack_synthetic.py \
#     --img 1280 \
#     --batch-size 8 \
#     --data mapillary_vistas.yaml \
#     --weights /home/chawins/adv-patch-bench/yolov5/runs/train/exp2/weights/best.pt \
#     --exist-ok \
#     --workers 8 \
#     --task train \
#     --save-exp-metrics \
#     --num-bg 16 \
#     --plot-octagons \
#     --load-patch ./adv_patch.pkl \
#     --synthetic

# CUDA_VISIBLE_DEVICES=0 python val_attack_synthetic.py \
#     --img 1280 \
#     --batch-size 8 \
#     --data mapillary_vistas.yaml \
#     --weights /home/chawins/adv-patch-bench/yolov5/runs/train/exp2/weights/best.pt \
#     --exist-ok \
#     --workers 8 \
#     --task train \
#     --save-exp-metrics \
#     --num-bg 16 \
#     --load-patch arrow \
#     --plot-octagons \
#     --apply-patch
