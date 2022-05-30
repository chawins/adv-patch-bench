# adv-patch-bench

## Dependencies

Test with

- `python == 3.8`
- `cuda >= 11.2`
- `kornia == 0.6.3`: Using version `>= 0.6.4` will raise an error.

```bash
conda install -y pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
conda install -y scipy pandas scikit-learn pip seaborn
conda upgrade -y numpy scipy pandas scikit-learn
conda install -y -c conda-forge opencv albumentations timm kornia==0.6.3
python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'
```

## Dataset

### MTSD

- MTSD: [link](https://www.mapillary.com/dataset/trafficsign)
- Have not found a way to automatically download the dataset.
- `prep_mtsd_for_yolo.py`: Prepare MTSD dataset for YOLOv5.
- YOLO expects samples and labels in `root_dir/images/*` and `root_dir/labels/*`, respectively. See [this link](https://github.com/ultralytics/yolov5/wiki/Train-Custom-Data#13-organize-directories) for more detail.
- Training set: MTSD training. Symlink to `~/data/yolo_data/(images or labels)/train`.
- Validation set: MTSD validation Symlink to `~/data/yolo_data/(images or labels)/val`.
- Test set: Combine Vistas training and validation. Symlink to `~/data/yolo_data/(images or labels)/test`.
- If you run into `Argument list too long` error, try to raise limit of argument stack size by `ulimit -S -s 100000000`. [link](https://unix.stackexchange.com/a/401797)

```bash
# Prepare MTSD dataset
# Dataset should be extracted to ~/data/mtsd_v2_fully_annotated
python prep_mtsd_for_yolo.py
python prep_mtsd_for_detectron.py
# FIXME: change yolo_data
LABEL_NAME=labels_no_color
cd ~/data/ && mkdir yolo_data && mkdir yolo_data/images yolo_data/labels

cd ~/data/yolo_data/images/
ln -s ~/data/mtsd_v2_fully_annotated/images/train train
ln -s ~/data/mtsd_v2_fully_annotated/images/val val
cd ~/data/yolo_data/labels/
ln -s ~/data/mtsd_v2_fully_annotated/$LABEL_NAME/train train
ln -s ~/data/mtsd_v2_fully_annotated/$LABEL_NAME/val val
```

<!-- ### Cityscapes

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
- Use API at this [link](https://panoptic-parts.readthedocs.io/en/stable/api_and_code.html#visualization) to visualize the labels. -->

### Mapillary

- `prep_mapillary.py`: Prepare Vistas dataset for YOLOv5 using a pretrained classifier to determine classes of the signs. May require substantial memory to run. Insufficient memory can lead to the script getting killed with no error message.

```bash
# Dataset should be extracted to ~/data/mapillary_vistas (use symlink if needed)
CUDA_VISIBLE_DEVICES=0 python prep_mapillary.py --split train --resume PATH_TO_CLASSIFIER
CUDA_VISIBLE_DEVICES=0 python prep_mapillary.py --split val --resume PATH_TO_CLASSIFIER

# Combined train and val partition into "combined"
cd ~/data/mapillary_vistas
mkdir no_color && cd no_color
mkdir combined && cd combined
mkdir images labels detectron_labels
ln -s ~/data/mapillary_vistas/training/images/* images/
ln -s ~/data/mapillary_vistas/validation/images/* images/
ln -s ~/data/mapillary_vistas/training/labels_no_color/* labels/
ln -s ~/data/mapillary_vistas/validation/labels_no_color/* labels/
ln -s ~/data/mapillary_vistas/training/detectron_labels_no_color/* detectron_labels/
ln -s ~/data/mapillary_vistas/validation/detectron_labels_no_color/* detectron_labels/
```

## YOLOv5

- Install required packages: `pip install -r requirements.txt`

## YOLOR

```bash
# Download yolor-p6 pretrained weights
cd ./yolor/scripts && gdown 1Tdn3yqpZ79X7R1Ql0zNlNScB1Dv9Fp76
```

## Training

- Extremely small objects are filtered out by default by YOLO. This is in `utils/autoanchor.py` on line 120.
- We use the default training hyperparameters: `hyp.scratch.yaml`.
- The pretrained model is trained on COCO training set.
- 2 V100 GPUs with 24 CPU cores: ~20 mins/epoch

```bash
sh train_yolo.sh
```

<!-- ## Other Tips -->
