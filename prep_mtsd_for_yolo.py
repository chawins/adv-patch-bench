import json
import os
import pdb
from os.path import expanduser, join
import shutil

import pandas as pd
from tqdm import tqdm

from hparams import (MIN_OBJ_AREA, NUM_CLASSES, TS_COLOR_DICT,
                     TS_COLOR_OFFSET_DICT)


def readlines(path):
    with open(path, 'r') as f:
        lines = f.readlines()
    return [line.strip() for line in lines]

# path = expanduser('~/data/mtsd_v2_fully_annotated/')
# csv_path = expanduser('~/adv-patch-bench/traffic_sign_dimension_v6.csv')
# anno_path = expanduser(join(path, 'annotations'))
# label_path = expanduser(join(path, 'labels'))
# data = pd.read_csv(csv_path)

path = '/data/shared/mtsd_v2_fully_annotated/'
csv_path = '/data/shared/mtsd_v2_fully_annotated/traffic_sign_dimension_v6.csv'
similarity_df_csv_path = 'similar_files_df.csv'
anno_path = join(path, 'annotations')
label_path = join(path, 'labels')
data = pd.read_csv(csv_path)
similar_files_df = pd.read_csv(similarity_df_csv_path)
use_mtsd_original_labels = False

selected_labels = list(TS_COLOR_OFFSET_DICT.keys())
mtsd_label_to_class_index = {}
for idx, row in data.iterrows():
    if row['target'] in TS_COLOR_OFFSET_DICT and not use_mtsd_original_labels:
        idx = TS_COLOR_OFFSET_DICT[row['target']]
        color_list = TS_COLOR_DICT[row['target']]
        # print(row['sign'], row['target'])
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
print(f'Found {len(json_files)} files')

<<<<<<< HEAD
similar_files_count = 0
=======
num_too_small = 0
num_other = 0

>>>>>>> b72b781... add message for prep_mtsd
for json_file in tqdm(json_files):
    filename = json_file.split('.')[-2].split('/')[-1]

    jpg_filename = f'{filename}.jpg'
    if jpg_filename in similar_files_df['filename'].values:
        similar_files_count += 1
        # qqq
        continue
    
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

        class_index = mtsd_label_to_class_index.get(obj['label'], bg_idx)
        # Compute object area if the image were to be resized to have width of 1280 pixels
        # obj_area = (obj_width * 1280) * (height / width * 1280)
        obj_area = (obj_width * 1280) * (obj_height * height / width * 1280)
        # Remove labels for small or "other" objects
        if obj_area < MIN_OBJ_AREA:
            num_too_small += 1
            continue
        if class_index == NUM_CLASSES - 1:
            num_other += 1
            continue
        text += f'{class_index} {x_center} {y_center} {obj_width} {obj_height} 0\n'
    
    with open(join(label_path, split, filename + '.txt'), 'w') as f:
        f.write(text)

<<<<<<< HEAD
print('[INFO] there are', similar_files_count, 'similar files in mapillary and mtsd')

print('[INFO] removing images in both the mapillary dataset and mtsd from mtsd')


data_path = join(path, 'images/')
new_data_path = join(path, 'images_mapillary_duplicates/')
for split in splits:
    os.makedirs(os.path.join(new_data_path, split), exist_ok=True)

for json_file in tqdm(json_files):
    filename = json_file.split('.')[-2].split('/')[-1]
    jpg_filename = f'{filename}.jpg'
    split = split_dict[filename]
    if jpg_filename not in similar_files_df['filename'].values:
        continue
    image_path = os.path.join(data_path, split, jpg_filename)
    image_new_path = os.path.join(new_data_path, split, jpg_filename)
    shutil.move(image_path, image_new_path)
=======
print(f'{num_too_small} signs are too small, and {num_other} of the remaining ones have "other" class.')
print('Finished.')
>>>>>>> b72b781... add message for prep_mtsd
