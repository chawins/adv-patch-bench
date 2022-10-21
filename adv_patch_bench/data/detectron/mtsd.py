"""Register and load MTSD dataset."""

import json
import pathlib
from typing import Any, Dict, List, Optional

import pandas as pd
from adv_patch_bench.utils.types import DetectronSample
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.structures import BoxMode
from hparams import (
    DEFAULT_PATH_MTSD_LABEL,
    PATH_SIMILAR_FILES,
    TS_COLOR_DICT,
    TS_COLOR_OFFSET_DICT,
)
from tqdm import tqdm

_ALLOWED_SPLITS = ("train", "test", "val")


def _readlines(path: str) -> List:
    with open(path, "r") as f:
        lines = f.readlines()
    return [line.strip() for line in lines]


def get_mtsd_anno(
    base_path: str,
    use_color: bool,
    use_mtsd_original_labels: bool,
    class_names: List[str],
) -> Dict[str, Any]:
    """Get MTSD annotation and metadata needed for loading dataset."""
    label_path: pathlib.Path = pathlib.Path(base_path) / "annotations"
    similarity_df_csv_path: str = PATH_SIMILAR_FILES
    similar_files_df: pd.DataFrame = pd.read_csv(similarity_df_csv_path)

    # Load annotation file that contains dimension and shape of each MTSD label
    label_map: pd.DataFrame = pd.read_csv(DEFAULT_PATH_MTSD_LABEL)
    # Collect mapping from original MTSD labels to new class index
    mtsd_label_to_class_index: Dict[str, int] = {}
    for idx, row in label_map.iterrows():
        if use_mtsd_original_labels:
            mtsd_label_to_class_index[row["sign"]] = idx
        elif any([row["target"] in c for c in class_names]):
            if use_color:
                cat_idx = TS_COLOR_OFFSET_DICT[row["target"]]
                color_list = TS_COLOR_DICT[row["target"]]
                if len(color_list) > 0:
                    cat_idx += color_list.index(row["color"])
            else:
                cat_idx = class_names.index(row["target"])
            mtsd_label_to_class_index[row["sign"]] = cat_idx

    # Get all JSON files
    json_files: List[str] = [
        str(f)
        for f in label_path.iterdir()
        if f.is_file() and f.suffix == ".json"
    ]

    return {"similar_files_df": similar_files_df,
            "mtsd_label_to_class_index": mtsd_label_to_class_index,
            "json_files": json_files}


def get_mtsd_dict(
    split: str,
    data_path: str,
    json_files: Optional[List[str]] = None,
    similar_files_df: Optional[pd.DataFrame] = None,
    mtsd_label_to_class_index: Optional[Dict[str, int]] = None,
    bg_class_id: int = 10,
    ignore_bg_class: bool = False,
) -> List[DetectronSample]:
    """Get MTSD dataset as list of samples in Detectron2 format.

    Args:
        split: Dataset split to consider.
        data_path: Base path to dataset. Defaults to "~/data/".
        json_files: List of paths to JSON files each of which contains original
            annotation of one image.
        similar_files_df: DataFrame of duplicated files between MTSD and
            Mapillary Vistas.
        mtsd_label_to_class_index: Dictionary that maps original MTSD labels to
            new class index.
        bg_class_id: Background class index.
        ignore_bg_class: Whether to ignore background class (last class index).
            Defaults to False.

    Raises:
        ValueError: split is not among _ALLOWED_SPLITS.

    Returns:
        List of MTSD samples in Detectron2 format.
    """
    if split not in _ALLOWED_SPLITS:
        raise ValueError(
            f"split must be among {_ALLOWED_SPLITS}, but it is {split}!"
        )

    if (json_files is None or similar_files_df is None or mtsd_label_to_class_index is None):
        raise ValueError(
            "Some MTSD metadata are missing! Please use get_mtsd_anno() to get "
            "all the neccessary metadata."
        )

    dpath: pathlib.Path = pathlib.Path(data_path)
    filenames = _readlines(str(dpath / "splits" / split + ".txt"))
    filenames = set(filenames)
    dataset_dicts: List[DetectronSample] = []

    for idx, json_file in tqdm(enumerate(json_files)):

        filename: str = json_file.split("/")[-1].split(".")[0]
        # Skip samples not in this split
        if filename not in filenames:
            continue
        jpg_filename: str = f"{filename}.jpg"
        # Skip samples that appear in Mapillary Vistas
        if jpg_filename in similar_files_df["filename"].values:
            continue

        # Read JSON files
        with open(json_file) as f:
            anno: Dict[str, Any] = json.load(f)

        height, width = anno["height"], anno["width"]
        record: DetectronSample = {
            "file_name": str(dpath / split / jpg_filename),
            "image_id": idx,
            "width": width,
            "height": height,
        }

        # Populate record or sample with its objects
        objs: List[Dict[str, Any]] = []
        for obj in anno["objects"]:
            class_index: int = mtsd_label_to_class_index.get(
                obj["label"], bg_class_id
            )
            # Remove labels for small or "other" objects
            if ignore_bg_class and class_index == bg_class_id:
                continue
            obj: Dict[str, Any] = {
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
    base_path: str = "~/data/",
    use_color: bool = False,
    use_mtsd_original_labels: bool = False,
    ignore_bg_class: bool = False,
    class_names: List[str] = ["circle"],
) -> None:
    """Register MTSD dataset on Detectron2.

    Args:
        base_path: Base path to dataset. Defaults to "~/data/".
        use_color: Whether sign color is used for grouping MTSD labels.
        use_mtsd_original_labels: Whether to use original MTSD labels instead of
            REAP annotations. Default to False.
        ignore_bg_class: Whether to ignore background class (last class index).
            Defaults to False.
        class_names: List of class names. Defaults to ["circle"].
    """

    mtsd_anno: Dict[str, Any] = get_mtsd_anno(
        base_path, use_color, use_mtsd_original_labels, class_names
    )
    label_map: pd.DataFrame = mtsd_anno["label_map"]
    bg_class_id: int = len(class_names) - 1

    for split in _ALLOWED_SPLITS:
        DatasetCatalog.register(
            f"mtsd_{split}",
            lambda s=split: get_mtsd_dict(
                split=s,
                data_path=base_path,
                bg_class_id=bg_class_id,
                ignore_bg_class=ignore_bg_class,
                **mtsd_anno,
            ),
        )
        if use_mtsd_original_labels:
            thing_classes = label_map["sign"].tolist()
        else:
            thing_classes = class_names
            if ignore_bg_class:
                thing_classes = thing_classes[:-1]

        MetadataCatalog.get(f"mtsd_{split}").set(thing_classes=thing_classes)
