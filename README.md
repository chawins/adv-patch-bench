# adv-patch-bench

Packages

```bash
conda install -c conda-forge opencv
```

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

Get MTSD data ready to train YOLOv5.

```bash
# Place the downloaded and extracted `mtsd_v2_fully_annotated` in ~/data/
python prep_mtsd_for_yolo.py
cd ~/data/mtsd_v2_fully_annotated
mkdir images && cd images
ln -s ../train/ train
ln -s ../test/ test
ln -s ../val/ val
```

TODO: Get Mapillary data ready for testing and the benchmark.

## YOLOv5

- Install required packages: `pip install -r requirements.txt`

## YOLOR

```bash
# Download yolor-p6 pretrained weights
cd ./yolor/scripts && gdown 1Tdn3yqpZ79X7R1Ql0zNlNScB1Dv9Fp76
```

### Data Preparation

- `prep_mtsd_for_yolo.py`: Prepare MTSD dataset for YOLOv5.
- `prep_vistas_for_yolo.py`: Prepare Vistas dataset for YOLOv5 using a pretrained classifier to determine classes of the signs. May require substantial memory to run. Insufficient memory can lead to the script getting killed with no error message.
- YOLO expects samples and labels in `root_dir/images/*` and `root_dir/labels/*`, respectively. See [this link](https://github.com/ultralytics/yolov5/wiki/Train-Custom-Data#13-organize-directories) for more detail.
- Training set: MTSD training. Symlink to `~/data/yolo_data/(images or labels)/train`.
- Validation set: MTSD validation Symlink to `~/data/yolo_data/(images or labels)/val`.
- Test set: Combine Vistas training and validation. Symlink to `~/data/yolo_data/(images or labels)/test`.
- If you run into `Argument list too long` error, try to raise limit of argument stack size by `ulimit -S -s 100000000`. [Ref.](https://unix.stackexchange.com/a/401797)

```bash
# Prepare MTSD path
# Dataset should be extracted to ~/data/mtsd_v2_fully_annotated
# FIXME: change yolo_data
LABEL_NAME=labels
cd ~/data/ && mkdir yolo_data && mkdir yolo_data/images yolo_data/labels
mkdir yolo_data/images/train yolo_data/images/val yolo_data/images/test
mkdir yolo_data/labels/train yolo_data/labels/val yolo_data/labels/test

cd ~/data/yolo_data/images/train
ln -s ~/data/mtsd_v2_fully_annotated/images/train/* .
cd ~/data/yolo_data/images/val
ln -s ~/data/mtsd_v2_fully_annotated/images/val/* .

cd ~/data/yolo_data/labels/train
ln -s ~/data/mtsd_v2_fully_annotated/$LABEL_NAME/train/* .
cd ~/data/yolo_data/labels/val
ln -s ~/data/mtsd_v2_fully_annotated/$LABEL_NAME/val/* .

# Prepare Mapillary path
# Dataset should be extracted to ~/data/mapillary_vistas
cd ~/data/mapillary_vistas
mkdir no_color && cd no_color
mkdir test val
cd ~/data/mapillary_vistas/no_color/test
ln -s ~/data/mapillary_vistas/training/images .
ln -s ~/data/mapillary_vistas/training/labels_no_color/ labels
cd ~/data/mapillary_vistas/no_color/val
ln -s ~/data/mapillary_vistas/validation/images .
ln -s ~/data/mapillary_vistas/validation/labels_no_color/ labels

# FIXME
# Change data path in mtsd.yml in adv-patch-bench/yolov5/data/ to the absolute
# path to yolo_data
```

### Training

- Extremely small objects are filtered out by default by YOLO. This is in `utils/autoanchor.py` on line 120.
- We use the default training hyperparameters: `hyp.scratch.yaml`.
- The pretrained model is trained on COCO training set.
- 2 V100 GPUs with 24 CPU cores: ~20 mins/epoch

```bash
sh train_yolo.sh
```

### Other Tips

- If you run into `Argument list too long` error when doing symlink. Try `ulimit -s 65536`.
