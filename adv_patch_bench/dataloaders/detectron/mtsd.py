import json
import os
from os.path import expanduser, join
from typing import Any, Dict, List, Tuple

import pandas as pd
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.structures import BoxMode
from hparams import (
    MIN_OBJ_AREA,
    OTHER_SIGN_CLASS,
    PATH_APB_ANNO,
    PATH_MTSD_BASE,
    PATH_SIMILAR_FILES,
    TS_COLOR_DICT,
    TS_COLOR_LABEL_LIST,
    TS_COLOR_OFFSET_DICT,
    TS_NO_COLOR_LABEL_LIST,
)
from tqdm import tqdm


def readlines(path: str) -> List:
    with open(path, "r") as f:
        lines = f.readlines()
    return [line.strip() for line in lines]


def get_mtsd_dict(
    split: str,
    path: str,
    json_files: List[str],
    similar_files_df: Any,
    mtsd_label_to_class_index: Dict,
    bg_idx: int,
    ignore_other: bool = False,
) -> List[Dict[str, Any]]:

    filenames = readlines(expanduser(join(path, "splits", split + ".txt")))
    filenames = set(filenames)
    dataset_dicts = []

    for idx, json_file in tqdm(enumerate(json_files)):

        filename = json_file.split(".")[-2].split("/")[-1]
        if filename not in filenames:
            continue
        jpg_filename = f"{filename}.jpg"
        if jpg_filename in similar_files_df["filename"].values:
            continue

        # Read JSON files
        with open(json_file) as f:
            anno = json.load(f)

        height, width = anno["height"], anno["width"]
        record = {
            "file_name": f'{path}/{split}/{json_file.split("/")[-1].split(".")[0]}.jpg',
            "image_id": idx,
            "width": width,
            "height": height,
        }
        objs = []
        for obj in anno["objects"]:
            class_index = mtsd_label_to_class_index.get(obj["label"], bg_idx)
            # Compute object area if the image were to be resized to have width of 1280 pixels
            obj_width = obj["bbox"]["xmax"] - obj["bbox"]["xmin"]
            obj_height = obj["bbox"]["ymax"] - obj["bbox"]["ymin"]
            # Scale by 1280 / width to normalize varying image size (this is not a bug)
            obj_area = (obj_width / width * 1280) * (obj_height / width * 1280)
            # Remove labels for small or "other" objects
            if obj_area < MIN_OBJ_AREA or (
                ignore_other and class_index == bg_idx
            ):
                continue
            obj = {
                "bbox": [
                    obj["bbox"]["xmin"],
                    obj["bbox"]["ymin"],
                    obj["bbox"]["xmax"],
                    obj["bbox"]["ymax"],
                ],
                "bbox_mode": BoxMode.XYXY_ABS,
                "category_id": class_index,
            }
            objs.append(obj)

        # Skip images with no object of interest
        if len(objs) == 0:
            continue

        record["annotations"] = objs
        dataset_dicts.append(record)

    return dataset_dicts


def register_mtsd(
    use_mtsd_original_labels: bool = False,
    use_color: bool = False,
    ignore_other: bool = False,
) -> Tuple[Any, ...]:
    path = PATH_MTSD_BASE
    csv_path = PATH_APB_ANNO
    similarity_df_csv_path = PATH_SIMILAR_FILES

    anno_path = expanduser(join(path, "annotations"))
    data = pd.read_csv(csv_path)
    similar_files_df = pd.read_csv(similarity_df_csv_path)

    if use_mtsd_original_labels:
        dataset = "mtsd_original"
    else:
        if use_color:
            label_dict = TS_COLOR_OFFSET_DICT
            dataset = "mtsd_color"
        else:
            label_dict = TS_COLOR_DICT
            dataset = "mtsd_no_color"
        selected_labels = list(label_dict.keys())

    mtsd_label_to_class_index = {}
    for idx, row in data.iterrows():
        if use_mtsd_original_labels:
            mtsd_label_to_class_index[row["sign"]] = idx
        elif row["target"] in label_dict:
            if use_color:
                cat_idx = TS_COLOR_OFFSET_DICT[row["target"]]
                color_list = TS_COLOR_DICT[row["target"]]
                if len(color_list) > 0:
                    cat_idx += color_list.index(row["color"])
            else:
                cat_idx = selected_labels.index(row["target"])
            mtsd_label_to_class_index[row["sign"]] = cat_idx
    bg_idx = OTHER_SIGN_CLASS[dataset]

    # Get all JSON files
    json_files = [
        join(anno_path, f)
        for f in os.listdir(anno_path)
        if os.path.isfile(join(anno_path, f)) and f.endswith(".json")
    ]

    splits = ["train", "test", "val"]
    for split in splits:
        DatasetCatalog.register(
            f"mtsd_{split}",
            lambda s=split: get_mtsd_dict(
                s,
                path,
                json_files,
                similar_files_df,
                mtsd_label_to_class_index,
                bg_idx,
                ignore_other=ignore_other,
            ),
        )
        if use_mtsd_original_labels:
            thing_classes = data["sign"].tolist()
        else:
            thing_classes = (
                TS_COLOR_LABEL_LIST if use_color else TS_NO_COLOR_LABEL_LIST
            )
            if ignore_other:
                thing_classes = thing_classes[:-1]
        MetadataCatalog.get(f"mtsd_{split}").set(thing_classes=thing_classes)

    return (
        path,
        json_files,
        similar_files_df,
        mtsd_label_to_class_index,
        bg_idx,
        ignore_other,
    )
