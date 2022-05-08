#!/bin/bash

# # Generate patch on synthetic dataset
# CUDA_VISIBLE_DEVICES=0 python -u generate_adv_patch.py \
#     --seed 0 \
#     --data mapillary_vistas.yaml \
#     --weights /data/shared/adv-patch-bench/yolov5/runs/train/exp2.1/weights/best.pt \
#     --patch-name stop_sign_synthetic_generated \
#     --csv-path mapillary_vistas_final_merged.csv \
#     --imgsz 1280 \
#     --padded_imgsz 992,1312 \
#     --obj-class 14 \
#     --obj-size 128 \
#     --obj-path attack_assets/octagon-915.0.png \
#     --num-bg 10 \
#     --bg-dir /data/shared/mtsd_v2_fully_annotated/train \
#     --save-images \
#     --generate-patch synthetic \
#     --attack-config-path attack_config.yaml

# # Test on synthetic dataset WITHOUT patch
# CUDA_VISIBLE_DEVICES=0 python -u val_attack_synthetic.py \
#     --imgsz 1280 \
#     --padded_imgsz 992,1312 \
#     --batch-size 8 \
#     --data mapillary_vistas.yaml \
#     --weights /data/shared/adv-patch-bench/yolov5/runs/train/exp2.1/weights/best.pt \
#     --exist-ok \
#     --workers 8 \
#     --task train \
#     --save-exp-metrics \
#     --metrics-confidence-threshold 0.359 \
#     --adv-patch-path ./runs/val/exp/stop_sign_synthetic_generated.pkl \
#     --tgt-csv-filepath mapillary_vistas_final_merged.csv \
#     --obj-class 14 \
#     --plot-class-examples 14 \
#     --interp bicubic \
#     --attack-type none \
#     --synthetic-eval \
#     --syn-obj-path attack_assets/octagon-915.0.png \
    
# # Test on synthetic dataset with patch
# CUDA_VISIBLE_DEVICES=0 python -u val_attack_synthetic.py \
#     --imgsz 1280 \
#     --padded_imgsz 992,1312 \
#     --batch-size 8 \
#     --data mapillary_vistas.yaml \
#     --weights /data/shared/adv-patch-bench/yolov5/runs/train/exp2.1/weights/best.pt \
#     --exist-ok \
#     --workers 8 \
#     --task train \
#     --save-exp-metrics \
#     --metrics-confidence-threshold 0.359 \
#     --adv-patch-path ./runs/val/exp/stop_sign_synthetic_generated.pkl \
#     --tgt-csv-filepath mapillary_vistas_final_merged.csv \
#     --obj-class 14 \
#     --plot-class-examples 14 \
#     --interp bicubic \
#     --attack-type synthetic \
#     --synthetic-eval \
#     --syn-obj-path attack_assets/octagon-915.0.png




# CUDA_VISIBLE_DEVICES=0 python -u val_attack_synthetic.py \
#     --imgsz 1280 \
#     --padded_imgsz 992,1312 \
#     --batch-size 8 \
#     --data mapillary_vistas.yaml \
#     --weights /data/shared/adv-patch-bench/yolov5/runs/train/exp2.1/weights/best.pt \
#     --exist-ok \
#     --workers 8 \
#     --task train \
#     --save-exp-metrics \
#     --tgt-csv-filepath mapillary_vistas_final_merged.csv \
#     --obj-class 14 \
#     --plot-class-examples 14 \
#     --interp bicubic \
#     --attack-type none 

# CUDA_VISIBLE_DEVICES=0 python -u val_attack_synthetic.py \
#     --imgsz 2560 \
#     --padded_imgsz 1952,2592 \
#     --batch-size 8 \
#     --data mapillary_vistas.yaml \
#     --weights /data/shared/adv-patch-bench/yolov5/runs/train/exp5/weights/best.pt \
#     --exist-ok \
#     --workers 8 \
#     --task train \
#     --save-exp-metrics \
#     --tgt-csv-filepath mapillary_vistas_final_merged.csv \
#     --obj-class 14 \
#     --plot-class-examples 14 \
#     --interp bicubic \
#     --attack-type none 

# CUDA_VISIBLE_DEVICES=0 python -u val_attack_synthetic.py \
#     --imgsz 1280 \
#     --padded_imgsz 992,1312 \
#     --batch-size 8 \
#     --data mapillary_vistas_no_color.yaml \
#     --weights /data/shared/adv-patch-bench/yolov5/runs/train/exp6/weights/best.pt \
#     --exist-ok \
#     --workers 8 \
#     --task train \
#     --save-exp-metrics \
#     --tgt-csv-filepath mapillary_vistas_final_merged.csv \
#     --obj-class 10 \
#     --plot-class-examples 10 \
#     --interp bicubic \
#     --attack-type none
    
# CUDA_VISIBLE_DEVICES=1 python -u val_attack_synthetic.py \
#     --imgsz 1280 \
#     --padded_imgsz 992,1312 \
#     --batch-size 8 \
#     --data mapillary_vistas_no_color.yaml \
#     --weights /data/shared/adv-patch-bench/yolov5/runs/train/exp7/weights/best.pt \
#     --exist-ok \
#     --workers 8 \
#     --task train \
#     --save-exp-metrics \
#     --tgt-csv-filepath mapillary_vistas_final_merged.csv \
#     --obj-class 10 \
#     --plot-class-examples 10 \
#     --interp bicubic \
#     --attack-type none 


# # Test real dataset WITHOUT patch
# CUDA_VISIBLE_DEVICES=0 python -u val_attack_synthetic.py \
#     --imgsz 1280 \
#     --padded_imgsz 992,1312 \
#     --batch-size 8 \
#     --data mapillary_vistas.yaml \
#     --weights /data/shared/adv-patch-bench/yolov5/runs/train/exp2.1/weights/best.pt \
#     --exist-ok \
#     --workers 8 \
#     --task train \
#     --save-exp-metrics \
#     --metrics-confidence-threshold 0.359 \
#     --tgt-csv-filepath mapillary_vistas_final_merged.csv \
#     --obj-class 14 \
#     --plot-class-examples 14 \
#     --interp bicubic \
#     --attack-type none \
#     --debug

# Test on real dataset with patch on train dataset
# CUDA_VISIBLE_DEVICES=0 python -u val_attack_synthetic.py \
#     --imgsz 1280 \
#     --padded_imgsz 992,1312 \
#     --batch-size 8 \
#     --data mapillary_vistas.yaml \
#     --weights /data/shared/adv-patch-bench/yolov5/runs/train/exp2.1/weights/best.pt \
#     --exist-ok \
#     --workers 8 \
#     --task train \
#     --save-exp-metrics \
#     --metrics-confidence-threshold 0.359 \
#     --adv-patch-path ./runs/val/exp/stop_sign_synthetic_generated.pkl \
#     --tgt-csv-filepath mapillary_vistas_final_merged.csv \
#     --obj-class 14 \
#     --plot-class-examples 14 \
#     --interp bicubic \
#     --attack-type real \
#     --min-area 100 

# # Test on real dataset with patch on val dataset
# CUDA_VISIBLE_DEVICES=0 python -u val_attack_synthetic.py \
#     --imgsz 1280 \
#     --padded_imgsz 992,1312 \
#     --batch-size 8 \
#     --data mapillary_vistas.yaml \
#     --weights /data/shared/adv-patch-bench/yolov5/runs/train/exp2.1/weights/best.pt \
#     --exist-ok \
#     --workers 8 \
#     --task val \
#     --save-exp-metrics \
#     --metrics-confidence-threshold 0.359 \
#     --adv-patch-path ./runs/val/exp/stop_sign_synthetic_generated.pkl \
#     --tgt-csv-filepath mapillary_vistas_validation_final_merged.csv \
#     --obj-class 14 \
#     --plot-class-examples 14 \
#     --interp bicubic \
#     --attack-type real \
#     --min-area 0 



# CUDA_VISIBLE_DEVICES=0 python -u val_attack_synthetic.py \
#     --imgsz 1280 \
#     --padded_imgsz 992,1312 \
#     --batch-size 8 \
#     --data mapillary_vistas.yaml \
#     --weights /data/shared/adv-patch-bench/yolov5/runs/train/exp2.1/weights/best.pt \
#     --exist-ok \
#     --workers 8 \
#     --task val \
#     --save-exp-metrics \
#     --metrics-confidence-threshold 0.359 \
#     --adv-patch-path ./runs/val/exp/stop_sign_synthetic_generated.pkl \
#     --tgt-csv-filepath mapillary_vistas_validation_final_merged.csv \
#     --obj-class 14 \
#     --plot-class-examples 14 \
#     --interp bicubic \
#     --attack-type per-sign \
#     --attack-config-path attack_config.yaml
    
# # Test on real dataset with patch (no transform)
# CUDA_VISIBLE_DEVICES=0 python -u val_attack_synthetic.py \
#     --imgsz 1280 \
#     --padded_imgsz 992,1312 \
#     --batch-size 8 \
#     --data mapillary_vistas.yaml \
#     --weights /data/shared/adv-patch-bench/yolov5/runs/train/exp2.1/weights/best.pt \
#     --exist-ok \
#     --workers 8 \
#     --task train \
#     --save-exp-metrics \
#     --metrics-confidence-threshold 0.359 \
#     --adv-patch-path ./runs/val/exp/stop_sign_synthetic_generated.pkl \
#     --tgt-csv-filepath mapillary_vistas_final_merged.csv \
#     --obj-class 14 \
#     --plot-class-examples 14 \
#     --interp bicubic \
#     --attack-type real \
#     --no-transform

# # Test on real dataset with patch (no relighting)
# CUDA_VISIBLE_DEVICES=0 python -u val_attack_synthetic.py \
#     --imgsz 1280 \
#     --padded_imgsz 992,1312 \
#     --batch-size 8 \
#     --data mapillary_vistas.yaml \
#     --weights /data/shared/adv-patch-bench/yolov5/runs/train/exp2.1/weights/best.pt \
#     --exist-ok \
#     --workers 8 \
#     --task train \
#     --save-exp-metrics \
#     --metrics-confidence-threshold 0.359 \
#     --adv-patch-path ./runs/val/exp/stop_sign_synthetic_generated.pkl \
#     --tgt-csv-filepath mapillary_vistas_final_merged.csv \
#     --obj-class 14 \
#     --plot-class-examples 14 \
#     --interp bicubic \
#     --attack-type real \
#     --no-relighting

# # Test on real dataset with patch (no transform and no relighting)
# CUDA_VISIBLE_DEVICES=0 python -u val_attack_synthetic.py \
#     --imgsz 1280 \
#     --padded_imgsz 992,1312 \
#     --batch-size 8 \
#     --data mapillary_vistas.yaml \
#     --weights /data/shared/adv-patch-bench/yolov5/runs/train/exp2.1/weights/best.pt \
#     --exist-ok \
#     --workers 8 \
#     --task train \
#     --save-exp-metrics \
#     --metrics-confidence-threshold 0.359 \
#     --adv-patch-path ./runs/val/exp/stop_sign_synthetic_generated.pkl \
#     --tgt-csv-filepath mapillary_vistas_final_merged.csv \
#     --obj-class 14 \
#     --plot-class-examples 14 \
#     --interp bicubic \
#     --attack-type real \
#     --no-transform \
#     --no-relighting

# CUDA_VISIBLE_DEVICES=0 python -u val_attack_synthetic.py \
#     --imgsz 1280 \
#     --padded_imgsz 992,1312 \
#     --batch-size 8 \
#     --data mapillary_vistas.yaml \
#     --weights /data/shared/adv-patch-bench/yolov5/runs/train/exp2.1/weights/best.pt \
#     --exist-ok \
#     --workers 8 \
#     --task train \
#     --save-exp-metrics \
#     --metrics-confidence-threshold 0.359 \
#     --adv-patch-path ./runs/val/exp/stop_sign_synthetic_generated.pkl \
#     --tgt-csv-filepath mapillary_vistas_final_merged.csv \
#     --obj-class 14 \
#     --plot-class-examples 14 \
#     --interp bicubic \
#     --attack-type per-sign \
#     --attack-config-path attack_config.yaml

# CUDA_VISIBLE_DEVICES=0 python -u val_attack_synthetic.py \
#     --imgsz 1280 \
#     --padded_imgsz 992,1312 \
#     --batch-size 8 \
#     --data mapillary_vistas.yaml \
#     --weights /data/shared/adv-patch-bench/yolov5/runs/train/exp2.1/weights/best.pt \
#     --exist-ok \
#     --workers 8 \
#     --task train \
#     --save-exp-metrics \
#     --metrics-confidence-threshold 0.359 \
#     --adv-patch-path ./runs/val/exp/stop_sign_synthetic_generated.pkl \
#     --tgt-csv-filepath mapillary_vistas_final_merged.csv \
#     --obj-class 14 \
#     --plot-class-examples 14 \
#     --interp bicubic \
#     --attack-type per-sign \
#     --attack-config-path attack_config.yaml \
#     --no-transform

# CUDA_VISIBLE_DEVICES=0 python -u val_attack_synthetic.py \
#     --imgsz 1280 \
#     --padded_imgsz 992,1312 \
#     --batch-size 8 \
#     --data mapillary_vistas.yaml \
#     --weights /data/shared/adv-patch-bench/yolov5/runs/train/exp2.1/weights/best.pt \
#     --exist-ok \
#     --workers 8 \
#     --task train \
#     --save-exp-metrics \
#     --metrics-confidence-threshold 0.359 \
#     --adv-patch-path ./runs/val/exp/stop_sign_synthetic_generated.pkl \
#     --tgt-csv-filepath mapillary_vistas_final_merged.csv \
#     --obj-class 14 \
#     --plot-class-examples 14 \
#     --interp bicubic \
#     --attack-type per-sign \
#     --attack-config-path attack_config.yaml \
#     --no-relighting

# CUDA_VISIBLE_DEVICES=0 python -u val_attack_synthetic.py \
#     --imgsz 1280 \
#     --padded_imgsz 992,1312 \
#     --batch-size 8 \
#     --data mapillary_vistas.yaml \
#     --weights /data/shared/adv-patch-bench/yolov5/runs/train/exp2.1/weights/best.pt \
#     --exist-ok \
#     --workers 8 \
#     --task train \
#     --save-exp-metrics \
#     --metrics-confidence-threshold 0.359 \
#     --adv-patch-path ./runs/val/exp/stop_sign_synthetic_generated.pkl \
#     --tgt-csv-filepath mapillary_vistas_final_merged.csv \
#     --obj-class 14 \
#     --plot-class-examples 14 \
#     --interp bicubic \
#     --attack-type per-sign \
#     --attack-config-path attack_config.yaml \
#     --no-transform \
#     --no-relighting















# CUDA_VISIBLE_DEVICES=0 python -u generate_adv_patch.py \
#     --seed 0 \
#     --data mapillary_vistas.yaml \
#     --weights /data/shared/adv-patch-bench/yolov5/runs/train/exp3/weights/best.pt \
#     --patch-name stop_sign_synthetic_generated \
#     --csv-path mapillary_vistas_final_merged.csv \
#     --obj-class 14 \
#     --obj-size 128 \
#     --obj-path attack_assets/octagon-915.0.png \
#     --num-bg 1 \
#     --bg-dir /data/shared/mtsd_v2_fully_annotated/train \
#     --save-images \
#     --generate-patch real \
#     --attack-config-path attack_config.yaml \
#     --imgsz 2560 \
#     --padded_imgsz 1952,2592 \
#     --imgsz 1280 \
#     --padded_imgsz 992,1312

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
#     --obj-size 128 \
#     --metrics-confidence-threshold 0.359 \
#     --apply-patch \
#     --load-patch ./runs/val/exp/stop_sign_transform.pkl \
#     --per-sign-attack \
#     --tgt-csv-filepath mapillary_vistas_final_merged.csv \
#     --attack-config-path attack_config.yaml \
#     --obj-class 14 \
#     --plot-class-examples 14

# CUDA_VISIBLE_DEVICES=1 python -u val_attack_synthetic.py \
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
#     --tgt-csv-filepath mapillary_vistas_final_merged.csv \
#     --obj-class 14 \
#     --plot-class-examples 14

# CUDA_VISIBLE_DEVICES=0 python -u val_attack_synthetic.py \
#     --imgsz 1280 \
#     --padded_imgsz 992,1312 \
#     --batch-size 8 \
#     --data mapillary_vistas.yaml \
#     --weights /data/shared/adv-patch-bench/yolov5/runs/train/exp2/weights/best.pt \
#     --exist-ok \
#     --workers 8 \
#     --task train \
#     --save-exp-metrics \
#     --obj-size 128 \
#     --metrics-confidence-threshold 0.359 \
#     --apply-patch \
#     --load-patch ./runs/val/exp/stop_sign_transform.pkl \
#     --synthetic \
#     --tgt-csv-filepath mapillary_vistas_final_merged.csv \
#     --attack-config-path attack_config.yaml \
#     --obj-class 14 \
#     --plot-class-examples 14 \
#     --interp bicubic
    













# CUDA_VISIBLE_DEVICES=0 python -u val_attack_synthetic.py \
#     --imgsz 1280 \
#     --padded_imgsz 992,1312 \
#     --batch-size 8 \
#     --data mapillary_vistas.yaml \
#     --weights /data/shared/adv-patch-bench/yolov5/runs/train/exp2/weights/best.pt \
#     --exist-ok \
#     --workers 8 \
#     --task train \
#     --save-exp-metrics \
#     --obj-size 128 \
#     --metrics-confidence-threshold 0.359 \
#     --tgt-csv-filepath mapillary_vistas_final_merged.csv \
#     --obj-class 14 \
#     --plot-class-examples 14 

# CUDA_VISIBLE_DEVICES=0 python -u val_attack_synthetic.py \
#     --imgsz 1280 \
#     --padded_imgsz 992,1312 \
#     --batch-size 8 \
#     --data mapillary_vistas.yaml \
#     --weights /data/shared/adv-patch-bench/yolov5/runs/train/exp2/weights/best.pt \
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
#     --interp bicubic
    









#     --apply-patch \
#     --load-patch ./runs/val/exp/stop_sign_transform.pkl \
#     --per-sign-attack \
#     --tgt-csv-filepath mapillary_vistas_final_merged.csv \
#     --attack-config-path attack_config.yaml \
#     --obj-class 14 \
#     --plot-class-examples 14

# CUDA_VISIBLE_DEVICES=0 python -u val_attack_synthetic.py \
#     --imgsz 1280 \
#     --padded_imgsz 992,1312 \
#     --batch-size 8 \
#     --data mapillary_vistas.yaml \
#     --weights /data/shared/adv-patch-bench/yolov5/runs/train/exp2/weights/best.pt \
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
#     --attack-config-path attack_config_1.yaml \
#     --obj-class 14 \
#     --plot-class-examples 14

# CUDA_VISIBLE_DEVICES=0 python -u val_attack_synthetic.py \
#     --imgsz 1280 \
#     --padded_imgsz 992,1312 \
#     --batch-size 8 \
#     --data mapillary_vistas.yaml \
#     --weights /data/shared/adv-patch-bench/yolov5/runs/train/exp2/weights/best.pt \
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
#     --attack-config-path attack_config_2.yaml \
#     --obj-class 14 \
#     --plot-class-examples 14

# CUDA_VISIBLE_DEVICES=0 python -u val_attack_synthetic.py \
#     --imgsz 1280 \
#     --padded_imgsz 992,1312 \
#     --batch-size 8 \
#     --data mapillary_vistas.yaml \
#     --weights /data/shared/adv-patch-bench/yolov5/runs/train/exp2/weights/best.pt \
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
#     --attack-config-path attack_config_3.yaml \
#     --obj-class 14 \
#     --plot-class-examples 14

    # --img-txt-path ./runs/successful_attack_filenames.txt \
    # --run-only-img-txt
    
# Generate patch on real dataset
CUDA_VISIBLE_DEVICES=1 python -u generate_adv_patch.py \
    --seed 0 \
    --data mapillary_vistas_no_color.yaml \
    --weights /data/shared/adv-patch-bench/yolov5/runs/train/exp2/weights/best.pt \
    --patch-name stop_sign \
    --task train \
    --csv-path mapillary_vistas_final_merged.csv \
    --imgsz 1280 \
    --padded_imgsz 992,1312 \
    --obj-class 14 \
    --obj-size 128 \
    --num-bg 1 \
    --bg-dir /data/shared/mtsd_v2_fully_annotated/train \
    --syn-obj-path attack_assets/octagon-915.0.png \
    --save-images \
    --generate-patch real \
    --attack-config-path attack_config.yaml \
    --patch_size 10

## Test patch on real dataset
# CUDA_VISIBLE_DEVICES=0 python -u val_attack_synthetic.py \
#     --imgsz 1280 \
#     --padded_imgsz 992,1312 \
#     --batch-size 8 \
#     --data mapillary_vistas.yaml \
#     --weights /data/shared/adv-patch-bench/yolov5/runs/train/exp2/weights/best.pt \
#     --exist-ok \
#     --workers 8 \
#     --task train \
#     --save-exp-metrics \
#     --obj-size 128 \
#     --metrics-confidence-threshold 0.359 \
#     --apply-patch \
#     --load-patch ./runs/val/exp5/stop_sign_synthetic_generated.pkl \
#     --tgt-csv-filepath mapillary_vistas_final_merged.csv \
#     --obj-class 14 \
#     --plot-class-examples 14 \
#     --interp bicubic

## Test patch on real dataset
# CUDA_VISIBLE_DEVICES=0 python -u val_attack_synthetic.py \
#     --imgsz 1280 \
#     --padded_imgsz 992,1312 \
#     --batch-size 8 \
#     --data mapillary_vistas.yaml \
#     --weights /data/shared/adv-patch-bench/yolov5/runs/train/exp2/weights/best.pt \
#     --exist-ok \
#     --workers 8 \
#     --task train \
#     --save-exp-metrics \
#     --obj-size 128 \
#     --metrics-confidence-threshold 0.359 \
#     --apply-patch \
#     --load-patch ./runs/val/exp2/stop_sign_synthetic_generated.pkl \
#     --tgt-csv-filepath mapillary_vistas_final_merged.csv \
#     --obj-class 14 \
#     --plot-class-examples 14 \
#     --interp bicubic \
#     --obj-path ./attack_assets/octagon-915.0.png \
#     --synthetic

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