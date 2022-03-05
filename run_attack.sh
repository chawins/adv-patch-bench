CUDA_VISIBLE_DEVICES=1 python -u val_attack_synthetic.py \
    --imgsz 1280 \
    --padded_imgsz 992,1312 \
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
    --load-patch ./runs/val/exp/stop_sign_synthetic_generated.pkl \
    --img-txt-path ./runs/successful_attack_filenames.txt \
    --run-only-img-txt
    
# CUDA_VISIBLE_DEVICES=1 python -u generate_adv_patch.py \
#     --seed 0 \
#     --data mapillary_vistas.yaml \
#     --weights /data/shared/adv-patch-bench/yolov5/runs/train/exp2/weights/best.pt \
#     --patch-name stop_sign_synthetic_generated \
#     --csv-path mapillary_vistas_final_merged.csv \
#     --imgsz 1280 \
#     --obj-class 14 \
#     --obj-size 128 \
#     --obj-path attack_assets/octagon-915.0.png \
#     --num-bg 20 \
#     --bg-dir /data/shared/mtsd_v2_fully_annotated/train \
#     --save-images \
#     --generate-patch real

# CUDA_VISIBLE_DEVICES=1 python -u val_attack_synthetic.py \
#     --imgsz 1280 \
#     --padded_imgsz 992,1312 \
#     --batch-size 8 \
#     --data mapillary_vistas.yaml \
#     --weights /data/shared/adv-patch-bench/yolov5/runs/train/exp2/weights/best.pt \
#     --exist-ok \
#     --workers 8 \
#     --task train \
#     --save-exp-metrics \
#     --plot-octagons \
#     --obj-size 128 \
#     --metrics-confidence-threshold 0.359
    
# CUDA_VISIBLE_DEVICES=1 python -u val_attack_synthetic.py \
#     --imgsz 1280 \
#     --padded_imgsz 992,1312 \
#     --batch-size 8 \
#     --data mapillary_vistas.yaml \
#     --weights /data/shared/adv-patch-bench/yolov5/runs/train/exp2/weights/best.pt \
#     --exist-ok \
#     --workers 8 \
#     --task train \
#     --save-exp-metrics \
#     --plot-octagons \
#     --obj-size 128 \
#     --metrics-confidence-threshold 0.359 \
#     --apply-patch \
#     --load-patch ./runs/val/exp/stop_sign_synthetic_generated.pkl \
#     --patch-loc 87 47 
    
# CUDA_VISIBLE_DEVICES=1 python -u val_attack_synthetic.py \
#     --imgsz 1280 \
#     --padded_imgsz 992,1312 \
#     --batch-size 8 \
#     --data mapillary_vistas.yaml \
#     --weights /data/shared/adv-patch-bench/yolov5/runs/train/exp2/weights/best.pt \
#     --exist-ok \
#     --workers 8 \
#     --task train \
#     --save-exp-metrics \
#     --plot-octagons \
#     --obj-size 128 \
#     --metrics-confidence-threshold 0.359 \
#     --apply-patch \
#     --load-patch ./runs/val/exp2/stop_sign_synthetic_generated.pkl \
#     --patch-loc 87 47 


# CUDA_VISIBLE_DEVICES=1 python -u val_attack_synthetic.py \
#     --imgsz 1280 \
#     --padded_imgsz 992,1312 \
#     --batch-size 8 \
#     --data mapillary_vistas.yaml \
#     --weights /data/shared/adv-patch-bench/yolov5/runs/train/exp2/weights/best.pt \
#     --exist-ok \
#     --workers 8 \
#     --task train \
#     --save-exp-metrics \
#     --plot-octagons \
#     --obj-size 128 \
#     --metrics-confidence-threshold 0.359


    
# CUDA_VISIBLE_DEVICES=1 python -u val_attack_synthetic.py \
#     --imgsz 1280 \
#     --padded_imgsz 992,1312 \
#     --batch-size 8 \
#     --data mapillary_vistas.yaml \
#     --weights /data/shared/adv-patch-bench/yolov5/runs/train/exp2/weights/best.pt \
#     --exist-ok \
#     --workers 8 \
#     --task train \
#     --save-exp-metrics \
#     --plot-octagons \
#     --obj-size 128 \
#     --metrics-confidence-threshold 0.359 
        
# generate patch on synthetic signs with NO rescaling and NO relighting
# CUDA_VISIBLE_DEVICES=0 python -u generate_adv_patch.py \
#     --seed 0 \
#     --data mapillary_vistas.yaml \
#     --weights /data/shared/adv-patch-bench/yolov5/runs/train/exp2/weights/best.pt \
#     --patch-name stop_sign_synthetic_generated \
#     --imgsz 1280 \
#     --obj-class 14 \
#     --obj-size 128 \
#     --obj-path attack_assets/octagon-915.0.png \
#     --num-bg 50 \
#     --bg-dir /data/shared/mtsd_v2_fully_annotated/train \
#     --save-images \
#     --generate-patch synthetic

# # generate patch on synthetic signs with rescaling and NO relighting
# CUDA_VISIBLE_DEVICES=0 python -u generate_adv_patch.py \
#     --seed 0 \
#     --data mapillary_vistas.yaml \
#     --weights /data/shared/adv-patch-bench/yolov5/runs/train/exp2/weights/best.pt \
#     --patch-name stop_sign_synthetic_generated \
#     --imgsz 1280 \
#     --obj-class 14 \
#     --obj-size 128 \
#     --obj-path attack_assets/octagon-915.0.png \
#     --num-bg 50 \
#     --bg-dir /data/shared/mtsd_v2_fully_annotated/train \
#     --save-images \
#     --generate-patch synthetic \
#     --rescaling

# # generate patch on synthetic signs with NO rescaling and relighting
# generate patch on synthetic signs with rescaling and relighting
# CUDA_VISIBLE_DEVICES=0 python -u generate_adv_patch.py \
#     --seed 0 \
#     --data mapillary_vistas.yaml \
#     --weights /data/shared/adv-patch-bench/yolov5/runs/train/exp2/weights/best.pt \
#     --patch-name stop_sign_synthetic_generated \
#     --imgsz 1312 \
#     --obj-class 14 \
#     --obj-size 128 \
#     --obj-path attack_assets/octagon-915.0.png \
#     --num-bg 50 \
#     --bg-dir /data/shared/mtsd_v2_fully_annotated/train \
#     --save-images \
#     --generate-patch synthetic \
#     --relighting

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
#     --obj-size 128 \
#     --metrics-confidence-threshold 0.359 \
#     --apply-patch \
#     --load-patch ./runs/val/exp28/stop_sign_synthetic_generated_v2.pkl \
#     --synthetic \
#     --patch-loc 87 47

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
#     --obj-size 128 \
#     --metrics-confidence-threshold 0.359 \
#     --apply-patch \
#     --load-patch ./runs/val/exp29/stop_sign_synthetic_generated_v2.pkl \
#     --synthetic \
#     --patch-loc 87 47