import pdb
import os
from os.path import join, expanduser
import json
from tqdm import tqdm

path = '~/data/mtsd/'
anno_path = expanduser(join(path, 'annotations'))
label_path = expanduser(join(path, 'labels'))


def readlines(path):
    with open(path, 'r') as f:
        lines = f.readlines()
    return [line.strip() for line in lines]


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
        text += f'0 {x_center} {y_center} {obj_width} {obj_height}\n'

    with open(join(label_path, split, filename + '.txt'), 'w') as f:
        f.write(text)
