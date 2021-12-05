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

Get data ready to train YOLOv5.

```bash
python prep_mtsd_for_yolo.py
cd data/mtsd
mkdir images && cd images
ln -s ../train/ train
ln -s ../test/ test
ln -s ../val/ val
```
