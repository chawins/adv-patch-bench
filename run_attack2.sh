
# CUDA_VISIBLE_DEVICES=0 python -u val_attack_synthetic.py \
#     --imgsz 2560 \
#     --padded_imgsz 1952,2592 \
#     --batch-size 2 \
#     --data mapillary_vistas.yaml \
#     --weights /data/shared/adv-patch-bench/yolov5/runs/train/exp3/weights/best.pt \
#     --exist-ok \
#     --workers 6 \
#     --task train \
#     --save-exp-metrics \
#     --obj-size 128 \
#     --metrics-confidence-threshold 0.359 \
#     --apply-patch \
#     --load-patch ./runs/val/exp2/stop_sign_10x10.pkl \
#     --per-sign-attack \
#     --tgt-csv-filepath mapillary_vistas_final_merged.csv \
#     --attack-config-path attack_config.yaml \
#     --obj-class 14 \
#     --plot-class-examples 14 \
#     --interp bilinear

CUDA_VISIBLE_DEVICES=1 python -u val_attack_synthetic.py \
    --imgsz 2560 \
    --padded_imgsz 1952,2592 \
    --batch-size 2 \
    --data mapillary_vistas.yaml \
    --weights /data/shared/adv-patch-bench/yolov5/runs/train/exp3/weights/best.pt \
    --exist-ok \
    --workers 6 \
    --task train \
    --save-exp-metrics \
    --obj-size 128 \
    --metrics-confidence-threshold 0.359 \
    --apply-patch \
    --load-patch ./runs/val/exp2/stop_sign_10x10.pkl \
    --per-sign-attack \
    --tgt-csv-filepath mapillary_vistas_final_merged.csv \
    --attack-config-path attack_config.yaml \
    --obj-class 14 \
    --plot-class-examples 14 \
    --interp bicubic

# CUDA_VISIBLE_DEVICES=2 python -u val_attack_synthetic.py \
#     --imgsz 2560 \
#     --padded_imgsz 1952,2592 \
#     --batch-size 2 \
#     --data mapillary_vistas.yaml \
#     --weights /data/shared/adv-patch-bench/yolov5/runs/train/exp3/weights/best.pt \
#     --exist-ok \
#     --workers 8 \
#     --task train \
#     --save-exp-metrics \
#     --obj-size 128 \
#     --metrics-confidence-threshold 0.359 \
#     --apply-patch \
#     --load-patch ./runs/val/exp/stop_sign_transform.pkl \
#     --per-sign-attack \
#     --tgt-csv-filepath mapillary_vistas_final_merged.csv \
#     --attack-config-path attack_config.yaml \
#     --obj-class 14 \
#     --plot-class-examples 14 \
#     --interp nearest