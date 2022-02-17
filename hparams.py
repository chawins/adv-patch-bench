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
print(TS_COLOR_LABEL_LIST)
