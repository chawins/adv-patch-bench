# adv-patch-bench

## Dependencies

Tested with

- `python >= 3.8`
- `cuda >= 11.2`
- `kornia == 0.6.3`: Using version `>= 0.6.4` will raise an error.

We recommend creating a new conda environment with python 3.8 because `kornia` and `detectron2` seem to often mess up dependencies and result in a segmentation fault.

```[bash]
# Install pytorch normally
conda install -y pytorch torchvision torchaudio cudatoolkit=11.6 -c pytorch
conda install -y scipy pandas scikit-learn pip seaborn
conda upgrade -y numpy scipy pandas scikit-learn
conda install -y -c conda-forge timm kornia==0.6.3
pip install opencv albumentations
pip install 'git+https://github.com/facebookresearch/detectron2.git'
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

```[bash]
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

```[bash]
# Dataset should be extracted to ~/data/mapillary_vistas (use symlink if needed)
CUDA_VISIBLE_DEVICES=0 python prep_mapillary.py --split train --resume PATH_TO_CLASSIFIER
CUDA_VISIBLE_DEVICES=0 python prep_mapillary.py --split val --resume PATH_TO_CLASSIFIER

# Combined train and val partition into "combined"
BASE_DIR=~/data/mapillary_vistas
cd $BASE_DIR
mkdir no_color && cd no_color
mkdir combined && cd combined
mkdir images labels detectron_labels
ln -s $BASE_DIR/training/images/* images/
ln -s $BASE_DIR/validation/images/* images/
ln -s $BASE_DIR/training/labels_no_color/* labels/
ln -s $BASE_DIR/validation/labels_no_color/* labels/
ln -s $BASE_DIR/training/detectron_labels_no_color/* detectron_labels/
ln -s $BASE_DIR/validation/detectron_labels_no_color/* detectron_labels/
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

```[bash]
sh train_yolo.sh
```

## Running Attack

- `configs` contains attack config files and detectron (Faster R-CNN) config files.

<!-- ## Other Tips -->

- To run on annotated signs only (consistent with results in the paper), use flag `--annotated-signs-only`. For Detectron, the dataset cache has to be deleted before this option to really take effect.

## TODOs

- Change interpolation (`interp`) type to `Enum` instead of `str`.