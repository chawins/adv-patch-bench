from os.path import expanduser, join

# Set paths
PATH_MAPILLARY_ANNO = {
    'train': './mapillary_vistas_training_final_merged.csv',
    'val': './mapillary_vistas_validation_final_merged.csv',
    'combined': './mapillary_vistas_final_merged.csv',
}
PATH_MTSD_BASE = expanduser('~/data/mtsd_v2_fully_annotated/')
# PATH_MTSD_BASE = '/datadrive/data/mtsd_v2_fully_annotated/'
PATH_MAPILLARY_BASE = expanduser('~/data/mapillary_vistas/')
# PATH_MAPILLARY_BASE = '/datadrive/data/mapillary_vistas/'
PATH_APB_ANNO = expanduser('~/adv-patch-bench/traffic_sign_dimension_v6.csv')
PATH_SIMILAR_FILES = './similar_files_df.csv'

# TODO: move to args in the future
SAVE_DIR_DETECTRON = './detectron_output'


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
