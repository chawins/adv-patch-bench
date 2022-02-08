import json
import os
import pdb
from os.path import expanduser, join

import pandas as pd
from tqdm import tqdm

from .hparams import TS_COLOR_DICT, TS_COLOR_OFFSET_DICT


def readlines(path):
    with open(path, 'r') as f:
        lines = f.readlines()
    return [line.strip() for line in lines]


path = '/data/shared/mtsd_v2_fully_annotated/'
csv_path = '/data/shared/mtsd_v2_fully_annotated/traffic_sign_dimension_v6.csv'
# path = expanduser('~/data/mtsd_v2_fully_annotated/')
# csv_path = expanduser('~/adv-patch-bench/traffic_sign_dimension_v6.csv')
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

for json_file in tqdm(json_files):
    filename = json_file.split('.')[-2].split('/')[-1]
    split = split_dict[filename]

    # Read JSON files
    with open(json_file) as f:
        anno = json.load(f)

    text = ''
    width, height = anno['width'], anno['height']
    for obj in anno['objects']:
        x_center = (obj['bbox']['xmin'] + obj['bbox']['xmax']) / 2 / width
        y_center = (obj['bbox']['ymin'] + obj['bbox']['ymax']) / 2 / height
        obj_width = (obj['bbox']['xmax'] - obj['bbox']['xmin']) / width
        obj_height = (obj['bbox']['ymax'] - obj['bbox']['ymin']) / height
        # text += f'0 {x_center} {y_center} {obj_width} {obj_height}\n'
        class_index = mtsd_label_to_class_index.get(obj['label'], bg_idx)
        text += f'{class_index} {x_center} {y_center} {obj_width} {obj_height}\n'

    with open(join(label_path, split, filename + '.txt'), 'w') as f:
        f.write(text)