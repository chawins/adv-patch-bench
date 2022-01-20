# adv-patch-bench

## Dataset

### BDD100K

### Cityscapes

See instructions at [https://github.com/mcordts/cityscapesScripts](https://github.com/mcordts/cityscapesScripts).

```bash
python -m pip install cityscapesscripts
csDownload --help
# For visualizing panoptic segmentation
cd datasets
git clone https://github.com/pmeletis/panoptic_parts.git
export PYTHONPATH="${PYTHONPATH}:/home/chawin/adv-patch-bench/datasets/panoptic_parts"
```

- We use `leftImg8bit_trainvaltest.zip` for the raw images and `gtFinePanopticParts_trainval.zip` for segmentation labels.
- Use API at this [link](https://panoptic-parts.readthedocs.io/en/stable/api_and_code.html#visualization) to visualize the labels.

### Mapillary

- MTSD: [link](https://www.mapillary.com/dataset/trafficsign)
- Have not found a way to automatically download the dataset.

Get data ready to train YOLOv5.

```bash
python prep_mtsd_for_yolo.py
cd data/mtsd
mkdir images && cd images
ln -s ../train/ train
ln -s ../test/ test
ln -s ../val/ val

# ... go to yolov5 dir
sh run.sh
```

## YOLOv5

- YOLO expects samples and labels in `root_dir/images/*` and `root_dir/labels/*`, respectively. See [this link](https://github.com/ultralytics/yolov5/wiki/Train-Custom-Data#13-organize-directories) for more detail.
- Training set: MTSD training. Symlink to `~/data/yolo_data/(images or labels)/train`.
- Validation set: MTSD validation Symlink to `~/data/yolo_data/(images or labels)/val`.
- Test set: Combine Vistas training and validation. Symlink to `~/data/yolo_data/(images or labels)/test`.

```bash
cd ~/data/yolo_data/images/train
ln -s ~/data/mtsd_v2_fully_annotated/images/train/* .
cd ~/data/yolo_data/images/val
ln -s ~/data/mtsd_v2_fully_annotated/images/val/* .
cd ~/data/yolo_data/images/test
ln -s ~/data/mapillary_vistas/training/images/* .

cd ~/data/yolo_data/labels/train
ln -s ~/data/mtsd_v2_fully_annotated/labels_v2/train/* .
cd ~/data/yolo_data/labels/val
ln -s ~/data/mtsd_v2_fully_annotated/labels_v2/val/* .
cd ~/data/yolo_data/labels/test
ln -s ~/data/mapillary_vistas/training/labels_v2/* .
```

- `prep_mtsd_for_yolo.py`: Prepare MTSD dataset for YOLOv5.
- `prep_vistas_for_yolo.py`: Prepare Vistas dataset for YOLOv5 using a pretrained classifier to determine classes of the signs.

### Other Tips

- If you run into `Argument list too long` error when doing symlink. Try `ulimit -s 65536`.
