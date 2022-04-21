import json
import os
import pdb
import random
from os.path import expanduser, join

import cv2
import pandas as pd
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.structures import BoxMode
from detectron2.utils.visualizer import Visualizer
from tqdm import tqdm

from hparams import (MIN_OBJ_AREA, NUM_CLASSES, TS_COLOR_DICT,
                     TS_COLOR_LABEL_LIST, TS_COLOR_OFFSET_DICT)


def readlines(path):
    with open(path, 'r') as f:
        lines = f.readlines()
    return [line.strip() for line in lines]


path = expanduser('~/data/mtsd_v2_fully_annotated/')
csv_path = expanduser('~/adv-patch-bench/traffic_sign_dimension_v6.csv')
anno_path = expanduser(join(path, 'annotations'))
label_path = expanduser(join(path, 'labels'))
data = pd.read_csv(csv_path)
use_mtsd_original_labels = False  # TODO

similarity_df_csv_path = 'similar_files_df.csv'
similar_files_df = pd.read_csv(similarity_df_csv_path)

selected_labels = list(TS_COLOR_OFFSET_DICT.keys())
mtsd_label_to_class_index = {}
for idx, row in data.iterrows():
    if row['target'] in TS_COLOR_OFFSET_DICT and not use_mtsd_original_labels:
        idx = TS_COLOR_OFFSET_DICT[row['target']]
        color_list = TS_COLOR_DICT[row['target']]
        if len(color_list) > 0:
            idx += color_list.index(row['color'])
        mtsd_label_to_class_index[row['sign']] = idx
    elif use_mtsd_original_labels:
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


def get_mtsd_dict(split):

    filenames = readlines(expanduser(join(path, 'splits', split + '.txt')))
    filenames = set(filenames)
    dataset_dicts = []

    for idx, json_file in tqdm(enumerate(json_files)):

        filename = json_file.split('.')[-2].split('/')[-1]
        if filename not in filenames:
            continue
        jpg_filename = f'{filename}.jpg'
        if jpg_filename in similar_files_df['filename'].values:
            continue

        # Read JSON files
        with open(json_file) as f:
            anno = json.load(f)

        height, width = anno['height'], anno['width']
        record = {
            'file_name': f'{path}/{split}/{json_file.split("/")[-1].split(".")[0]}.jpg',
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
            if obj_area < MIN_OBJ_AREA or class_index == bg_idx:
                continue
            obj = {
                'bbox': [obj['bbox']['xmin'], obj['bbox']['ymin'], obj['bbox']['xmax'], obj['bbox']['ymax']],
                'bbox_mode': BoxMode.XYXY_ABS,
                'category_id': class_index,
            }
            objs.append(obj)

        record['annotations'] = objs
        dataset_dicts.append(record)

    return dataset_dicts


splits = ['train', 'test', 'val']
for split in splits:
    DatasetCatalog.register(f'mtsd_{split}', lambda s=split: get_mtsd_dict(s))
    if use_mtsd_original_labels:
        MetadataCatalog.get(f'mtsd_{split}').set(thing_classes=data['sign'].tolist())
    else:
        MetadataCatalog.get(f'mtsd_{split}').set(thing_classes=TS_COLOR_LABEL_LIST[:-1])


# DEBUG
# metadata = MetadataCatalog.get('mtsd_train')
# dataset_dicts = get_mtsd_dict('train')
# for d in random.sample(dataset_dicts, 10):
#     img = cv2.imread(d["file_name"])
#     visualizer = Visualizer(img[:, :, ::-1], metadata=metadata, scale=0.5)
#     out = visualizer.draw_dataset_dict(d)
#     out.save('test.png')
#     import pdb
#     pdb.set_trace()
