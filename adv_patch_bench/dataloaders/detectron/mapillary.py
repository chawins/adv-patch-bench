import os
from os.path import isfile, join
from typing import List, Tuple

import pandas as pd
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.structures import BoxMode
from hparams import (
    MIN_OBJ_AREA,
    OTHER_SIGN_CLASS,
    PATH_MAPILLARY_ANNO,
    PATH_MAPILLARY_BASE,
    TS_COLOR_LABEL_LIST,
    TS_NO_COLOR_LABEL_LIST,
)
from tqdm import tqdm


def readlines(path: str) -> List:
    with open(path, "r") as f:
        lines = f.readlines()
    return [line.strip() for line in lines]


def get_mapillary_dict(
    split: str,
    base_path: str,
    bg_idx: int,
    use_color: bool = False,
    ignore_other: bool = False,
    only_annotated: bool = True,
) -> List:

    mapillary_split = {
        "train": "training",
        "test": "validation",
        "combined": "combined",
    }[split]
    subdir = "color" if use_color else "no_color"
    anno_path = f"{base_path}/{subdir}/{mapillary_split}/detectron_labels"
    img_path = f"{base_path}/{subdir}/{mapillary_split}/images"

    dataset_dicts = []
    anno_files = [
        join(anno_path, f)
        for f in os.listdir(anno_path)
        if isfile(join(anno_path, f))
    ]
    anno_files = sorted(anno_files)

    csv_path = PATH_MAPILLARY_ANNO[split]
    df = pd.read_csv(csv_path)

    for idx, anno_file in tqdm(enumerate(anno_files)):

        filename = anno_file.split(".txt")[0].split("/")[-1]
        jpg_filename = f"{filename}.jpg"
        row = df[df["filename"] == jpg_filename]

        # Annotation text files
        with open(anno_file) as f:
            anno = f.readlines()
            anno = [a.strip() for a in anno]

        width = float(anno[0].split(",")[5])
        height = float(anno[0].split(",")[6])
        record = {
            "file_name": f"{img_path}/{jpg_filename}",
            "image_id": idx,
            "width": width,
            "height": height,
        }
        objs = []
        for obj in anno:
            class_index, xmin, ymin, xmax, ymax, _, _, obj_id = obj.split(",")
            xmin, ymin, xmax, ymax = [
                float(x) for x in [xmin, ymin, xmax, ymax]
            ]
            class_index, obj_id = int(class_index), int(obj_id)
            if only_annotated and not any(row["object_id"] == obj_id):
                # If we want results on annotated signs, we set the class of
                # the unannotated ones to "other" or background class
                class_index = bg_idx
            # Compute object area if the image were to be resized to have width of 1280 pixels
            obj_width = xmax - xmin
            obj_height = ymax - ymin
            # Scale by 1280 / width to normalize varying image size (this is not a bug)
            obj_area = (obj_width / width * 1280) * (obj_height / width * 1280)
            # Remove labels for small or "other" objects
            if obj_area < MIN_OBJ_AREA or (
                ignore_other and class_index == bg_idx
            ):
                continue
            assert class_index <= bg_idx, (
                "You may have prepared Mapillary data"
                "with color, but use_color is False here."
            )
            obj = {
                "bbox": [xmin, ymin, xmax, ymax],
                "bbox_mode": BoxMode.XYXY_ABS,
                "category_id": class_index,
                "object_id": obj_id,
            }
            objs.append(obj)

        # Skip images with no object of interest
        if len(objs) == 0:
            continue

        record["annotations"] = objs
        dataset_dicts.append(record)

    return dataset_dicts


def register_mapillary(
    use_color: bool = False,
    ignore_other: bool = False,
    only_annotated: bool = True,
) -> Tuple:
    base_path = PATH_MAPILLARY_BASE
    dataset = "mapillary_color" if use_color else "mapillary_no_color"
    bg_idx = OTHER_SIGN_CLASS[dataset]

    thing_classes = TS_COLOR_LABEL_LIST if use_color else TS_NO_COLOR_LABEL_LIST
    if ignore_other:
        thing_classes = thing_classes[:-1]

    splits = ["train", "val", "combined"]
    for split in splits:
        DatasetCatalog.register(
            f"mapillary_{split}",
            lambda s=split: get_mapillary_dict(
                s,
                base_path,
                bg_idx,
                use_color=use_color,
                ignore_other=ignore_other,
                only_annotated=only_annotated,
            ),
        )
        MetadataCatalog.get(f"mapillary_{split}").set(
            thing_classes=thing_classes
        )

    return base_path, bg_idx, use_color, ignore_other, only_annotated
