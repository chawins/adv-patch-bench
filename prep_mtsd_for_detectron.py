import json
import os
import pdb
from os.path import expanduser, join
from detectron2.data import DatasetCatalog, MetadataCatalog
import pandas as pd
from detectron2.structures import BoxMode
from tqdm import tqdm

from hparams import (MIN_OBJ_AREA, NUM_CLASSES, TS_COLOR_DICT,
                     TS_COLOR_OFFSET_DICT)


def readlines(path):
    with open(path, 'r') as f:
        lines = f.readlines()
    return [line.strip() for line in lines]


path = expanduser('~/data/mtsd_v2_fully_annotated/')
csv_path = expanduser('~/adv-patch-bench/traffic_sign_dimension_v6.csv')
anno_path = expanduser(join(path, 'annotations'))
label_path = expanduser(join(path, 'labels'))
data = pd.read_csv(csv_path)

selected_labels = list(TS_COLOR_OFFSET_DICT.keys())
mtsd_label_to_class_index = {}
for _, row in data.iterrows():
    if row['target'] in TS_COLOR_OFFSET_DICT:
        idx = TS_COLOR_OFFSET_DICT[row['target']]
        color_list = TS_COLOR_DICT[row['target']]
        # print(row['sign'], row['target'])
        if len(color_list) > 0:
            idx += color_list.index(row['color'])
        mtsd_label_to_class_index[row['sign']] = idx
bg_idx = max(list(mtsd_label_to_class_index.values())) + 1

# Save filenames and the data partition they belong to
splits = ['train', 'test', 'val']
split_dict = {}
for split in splits:
    os.makedirs(join(label_path, split), exist_ok=True)
    filenames = readlines(expanduser(join(path, 'splits', split + '.txt')))
    for name in filenames:
        split_dict[name] = split

# Get all JSON files
json_files = [join(anno_path, f) for f in os.listdir(anno_path)
              if os.path.isfile(join(anno_path, f)) and f.endswith('.json')]
print(f'Found {len(json_files)} files')

for idx, json_file in tqdm(enumerate(json_files)):
    filename = json_file.split('.')[-2].split('/')[-1]
    split = split_dict[filename]

    # Read JSON files
    with open(json_file) as f:
        anno = json.load(f)

    height, width = anno['height'], anno['width']
    record = {
        'file_name': json_file.split('/')[-1],
        'image_id': idx,
        'width': width,
        'height': height,
    }
    objs = []
    for obj in anno['objects']:
        class_index = mtsd_label_to_class_index.get(obj['label'], bg_idx)
        # Compute object area if the image were to be resized to have width of 1280 pixels
        obj_width = (obj['bbox']['xmax'] - obj['bbox']['xmin'])
        obj_height = (obj['bbox']['ymax'] - obj['bbox']['ymin'])
        # Scale by 1280 / width to normalize varying image size (this is not a bug)
        obj_area = (obj_width / width * 1280) * (obj_height / width * 1280)
        # Remove labels for small or "other" objects
        if obj_area < MIN_OBJ_AREA or class_index == NUM_CLASSES - 1:
            continue
        obj = {
            'bbox': [obj['bbox']['xmin'], obj['bbox']['ymin'], obj['bbox']['xmax'], obj['bbox']['ymax']],
            'bbox_mode': BoxMode.XYXY_ABS,
            'category_id': class_index,
        }
        objs.append(obj)

splits = ['train', 'test', 'val']
split_dict = {}
for split in splits:
    DatasetCatalog.register(f'mtsd_{split}', lambda d=d: get_balloon_dicts("balloon/" + d))
    MetadataCatalog.get(f'mtsd_{split}').set(thing_classes=list(TS_COLOR_DICT.keys()))
    
balloon_metadata = MetadataCatalog.get("balloon_train")

# for d in ["train", "val"]:
#     DatasetCatalog.register("balloon_" + d, lambda d=d: get_balloon_dicts("balloon/" + d))
#     MetadataCatalog.get("balloon_" + d).set(thing_classes=["balloon"])
# balloon_metadata = MetadataCatalog.get("balloon_train")

# def get_balloon_dicts(img_dir):
#     json_file = os.path.join(img_dir, "via_region_data.json")
#     with open(json_file) as f:
#         imgs_anns = json.load(f)

#     dataset_dicts = []
#     for idx, v in enumerate(imgs_anns.values()):
#         record = {}

#         filename = os.path.join(img_dir, v["filename"])
#         height, width = cv2.imread(filename).shape[:2]

#         record["file_name"] = filename
#         record["image_id"] = idx
#         record["height"] = height
#         record["width"] = width

#         annos = v["regions"]
#         objs = []
#         for _, anno in annos.items():
#             assert not anno["region_attributes"]
#             anno = anno["shape_attributes"]
#             px = anno["all_points_x"]
#             py = anno["all_points_y"]
#             poly = [(x + 0.5, y + 0.5) for x, y in zip(px, py)]
#             poly = [p for x in poly for p in x]

#             obj = {
#                 "bbox": [np.min(px), np.min(py), np.max(px), np.max(py)],
#                 "bbox_mode": BoxMode.XYXY_ABS,
#                 "segmentation": [poly],
#                 "category_id": 0,
#             }
#             objs.append(obj)
#         record["annotations"] = objs
#         dataset_dicts.append(record)
#     return dataset_dicts


