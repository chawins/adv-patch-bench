import json
import os
import pdb
from os.path import expanduser, join

import pandas as pd
from tqdm import tqdm


def readlines(path):
    with open(path, 'r') as f:
        lines = f.readlines()
    return [line.strip() for line in lines]


color_dict = {
    'circle-750.0': ['white', 'blue', 'red'],   # (1) white+red, (2) blue+white
    'triangle-900.0': ['white', 'yellow'],  # (1) white, (2) yellow
    'triangle_inverted-1220.0': [],   # (1) white+red
    'diamond-600.0': [],    # (1) white+yellow
    'diamond-915.0': [],    # (1) yellow
    'square-600.0': [],     # (1) blue
    'rect-458.0-610.0': ['white', 'other'],  # (1) chevron (also multi-color), (2) white
    'rect-762.0-915.0': [],  # (1) white
    'rect-915.0-1220.0': [],    # (1) white
    'pentagon-915.0': [],   # (1) yellow
    'octagon-915.0': [],    # (1) red
}
class_idx = {
    'circle-750.0': 0,   # (1) white+red, (2) blue+white
    'triangle-900.0': 3,  # (1) white, (2) yellow
    'triangle_inverted-1220.0': 5,   # (1) white+red
    'diamond-600.0': 6,    # (1) white+yellow
    'diamond-915.0': 7,    # (1) yellow
    'square-600.0': 8,     # (1) blue
    'rect-458.0-610.0': 9,  # (1) chevron (also multi-color), (2) white
    'rect-762.0-915.0': 11,  # (1) white
    'rect-915.0-1220.0': 12,    # (1) white
    'pentagon-915.0': 13,   # (1) yellow
    'octagon-915.0': 14,    # (1) red
}

path = '/data/shared/mtsd_v2_fully_annotated/'
csv_path = '/data/shared/mtsd_v2_fully_annotated/traffic_sign_dimension_v6.csv'
# path = expanduser('~/data/mtsd_v2_fully_annotated/')
# csv_path = expanduser('~/adv-patch-bench/traffic_sign_dimension_v6.csv')
anno_path = expanduser(join(path, 'annotations'))
label_path = expanduser(join(path, 'labels'))
data = pd.read_csv(csv_path)

selected_labels = list(class_idx.keys())
mtsd_label_to_class_index = {}
for _, row in data.iterrows():
    if row['target'] in class_idx:
        idx = class_idx[row['target']]
        color_list = color_dict[row['target']]
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
