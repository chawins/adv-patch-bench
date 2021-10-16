import json
import os
import random
from os.path import join

import numpy as np
import pandas as pd
from PIL import Image
from tqdm.auto import tqdm


def load_annotation(label_path, image_key):
    with open(join(label_path, '{:s}.json'.format(image_key)), 'r') as fid:
        anno = json.load(fid)
    return anno


def main(data_dir, mtsd_label_to_shape_index, dataset_name, pad=0.):
    bg_idx = max(list(mtsd_label_to_shape_index.values())) + 1
    images, labels, names = [], [], []

    for split in ['train', 'val']:

        img_path = join(data_dir, split)
        label_path = join(data_dir, 'annotations')

        image_keys, exts = [], []
        for entry in os.scandir(img_path):
            if (entry.path.endswith('.jpg') or entry.path.endswith('.png')) and entry.is_file():
                image_keys.append(entry.name.split('.')[0])
                exts.append(entry.name.split('.')[1])

        for image_key, ext in tqdm(zip(image_keys, exts)):
            anno = load_annotation(label_path, image_key)

            with Image.open(os.path.join(img_path, f'{image_key}.{ext}')) as img:
                img = img.convert('RGB')

            for index, obj in enumerate(anno['objects']):
                if obj['properties']['ambiguous']:
                    continue
                class_name = obj['label']
                shape_index = mtsd_label_to_shape_index.get(class_name, bg_idx)
                x1 = obj['bbox']['xmin']
                y1 = obj['bbox']['ymin']
                x2 = obj['bbox']['xmax']
                y2 = obj['bbox']['ymax']

                box_length = (1 + pad) * max((x2 - x1, y2 - y1))
                width_change = box_length - (x2 - x1)
                height_change = box_length - (y2 - y1)
                x1 = x1 - width_change / 2
                x2 = x2 + width_change / 2
                y1 = y1 - height_change / 2
                y2 = y2 + height_change / 2
                img_cropped = img.crop((x1, y1, x2, y2))
                images.append(img_cropped.resize((128, 128), resample=Image.BICUBIC))
                labels.append(shape_index)
                names.append(f'{image_key}_{index}.png')

            # DEBUG
            # if len(images) > 100:
            #     break

    print('Label distribution: ', np.unique(labels, return_counts=True))

    # Train and val split
    num_samples = len(images)
    num_train = int(0.9 * num_samples)
    idx = np.arange(num_samples)
    np.random.shuffle(idx)
    idx = {
        'train': idx[:num_train],
        'val': idx[num_train:]
    }

    for split in ['train', 'val']:
        save_dir = join(data_dir, dataset_name, split)
        for i in range(bg_idx + 1):
            os.makedirs(join(save_dir, str(i)), exist_ok=True)
        for i in tqdm(idx[split]):
            images[i].save(join(save_dir, str(labels[i]), names[i]))


if __name__ == '__main__':

    # Set the parameters
    seed = 0
    data_dir = '/data/shared/mtsd_v2_fully_annotated/'
    csv_path = '/data/shared/mtsd_v2_fully_annotated/traffic_sign_dimension_v5.csv'
    dataset_name = 'cropped_signs_v6'
    selected_labels = [
        'circle-750.0',
        'triangle-900.0',
        'triangle_inverted-1220.0',
        'diamond-600.0',
        'diamond-915.0',
        'square-600.0',
        'rect-458.0-610.0',
        'rect-762.0-915.0',
        'rect-915.0-1220.0',
        'pentagon-915.0',
        'octagon-915.0',
    ]
    pad = 0.

    np.random.seed(seed)
    random.seed(seed)
    data = pd.read_csv(csv_path)

    print(np.unique(list(data['target'])))

    mtsd_label_to_shape_index = {}
    for _, row in data.iterrows():
        if row['target'] in selected_labels:
            mtsd_label_to_shape_index[row['sign']] = selected_labels.index(row['target'])

    main(data_dir, mtsd_label_to_shape_index, dataset_name, pad=pad)
