from os.path import expanduser, join

# Set paths
PATH_MAPILLARY_ANNO = {
    'train': './mapillary_vistas_training_final_merged.csv',
    'val': './mapillary_vistas_validation_final_merged.csv',
    'combined': './mapillary_vistas_final_merged.csv',
}
PATH_MTSD_BASE = expanduser('~/data/mtsd_v2_fully_annotated/')
PATH_MAPILLARY_BASE = expanduser('~/data/mapillary_vistas/')
PATH_APB_ANNO = expanduser('./traffic_sign_dimension_v6.csv')
PATH_SIMILAR_FILES = './similar_files_df.csv'
PATH_SYN_OBJ = './attack_assets/'

# TODO: move to args in the future
SAVE_DIR_DETECTRON = './detectron_output/'
SAVE_DIR_YOLO = './runs/val/'

MTSD_VAL_LABEL_COUNTS_DICT = {
    'circle-750.0': 2999,
    'triangle-900.0': 711,
    'triangle_inverted-1220.0': 347,
    'diamond-600.0': 176,
    'diamond-915.0': 1278,
    'square-600.0': 287,
    'rect-458.0-610.0': 585,
    'rect-762.0-915.0': 117,
    'rect-915.0-1220.0': 135,
    'pentagon-915.0': 30,
    'octagon-915.0': 181,
    'other-0.0-0.0': 19241,
}
MTSD_VAL_TOTAL_LABEL_COUNTS = sum(MTSD_VAL_LABEL_COUNTS_DICT.values())

MAPILLARY_LABEL_COUNTS_DICT = {
    'circle-750.0': 18144,
    'triangle-900.0': 1473,
    'triangle_inverted-1220.0': 1961,
    'diamond-600.0': 1107,
    'diamond-915.0': 3539,
    'square-600.0': 1898,
    'rect-458.0-610.0': 1580,
    'rect-762.0-915.0': 839,
    'rect-915.0-1220.0': 638,
    'pentagon-915.0': 204,
    'octagon-915.0': 1001,
    'other-0.0-0.0': 60104,
}
MAPILLARY_TOTAL_LABEL_COUNTS = sum(MAPILLARY_LABEL_COUNTS_DICT.values())

# Counts of images where sign is present in
MAPILLARY_IMG_COUNTS_DICT = {
    'circle-750.0': 5325,
    'triangle-900.0': 548,
    'triangle_inverted-1220.0': 706,
    'diamond-600.0': 293,
    'diamond-915.0': 1195,
    'square-600.0': 729,
    'rect-458.0-610.0': 490,
    'rect-762.0-915.0': 401,
    'rect-915.0-1220.0': 333,
    'pentagon-915.0': 116,
    'octagon-915.0': 564,
    'other-0.0-0.0': 0,
}

# Traffic sign classes and colors
TS_COLOR_DICT = {
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
    'other-0.0-0.0': [],
}

# Generate dictionary of traffic sign class offset
TS_COLOR_OFFSET_DICT = {}
idx = 0
for k in TS_COLOR_DICT:
    TS_COLOR_OFFSET_DICT[k] = idx
    idx += max(1, len(TS_COLOR_DICT[k]))

# Generate dictionary of traffic sign class: name -> idx
TS_COLOR_LABEL_DICT = {}
idx = 0
for k in TS_COLOR_DICT:
    if len(TS_COLOR_DICT[k]) == 0:
        TS_COLOR_LABEL_DICT[f'{k}-none'] = idx
        idx += 1
    else:
        for color in TS_COLOR_DICT[k]:
            TS_COLOR_LABEL_DICT[f'{k}-{color}'] = idx
            idx += 1

# Make sure that ordering is correct
TS_COLOR_LABEL_LIST = list(TS_COLOR_LABEL_DICT.keys())
TS_NO_COLOR_LABEL_LIST = list(TS_COLOR_DICT.keys())
LABEL_LIST = {
    'mtsd_color': TS_COLOR_LABEL_LIST,
    'mapillary_color': TS_COLOR_LABEL_LIST,
    'mtsd_no_color': TS_NO_COLOR_LABEL_LIST,
    'mapillary_no_color': TS_NO_COLOR_LABEL_LIST,
}

# Get list of shape (no size, no color)
TS_SHAPE_LIST = list(set([l.split('-')[0] for l in TS_NO_COLOR_LABEL_LIST]))

MIN_OBJ_AREA = 0
NUM_CLASSES = len(TS_COLOR_LABEL_LIST)

DATASETS = ('mtsd_orig', 'mtsd_no_color', 'mtsd_color', 'mapillary_no_color',
            'mapillary_color')
OTHER_SIGN_CLASS = {
    'mtsd_orig': 89,
    'mtsd_no_color': len(TS_NO_COLOR_LABEL_LIST) - 1,
    'mtsd_color': len(TS_COLOR_LABEL_LIST) - 1,
    'mapillary_no_color': len(TS_NO_COLOR_LABEL_LIST) - 1,
    'mapillary_color': len(TS_COLOR_LABEL_LIST) - 1,
}

NUM_CLASSES = {
    'mtsd_orig': 401,
    'mtsd_no_color': len(TS_NO_COLOR_LABEL_LIST),
    'mtsd_color': len(TS_COLOR_LABEL_LIST),
    'mapillary_no_color': len(TS_NO_COLOR_LABEL_LIST),
    'mapillary_color': len(TS_COLOR_LABEL_LIST),
}