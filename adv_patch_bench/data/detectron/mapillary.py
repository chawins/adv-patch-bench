"""Register and load Mapillary Vistas dataset."""

import pathlib
from typing import Any, Dict, List, Optional

import pandas as pd
from adv_patch_bench.utils.types import DetectronSample
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.structures import BoxMode
from tqdm import tqdm

_ALLOWED_SPLITS = ("train", "test", "comgined")


def get_mapillary_dict(
    split: str,
    base_path: str,
    bg_class_id: int,
    ignore_bg_class: bool = False,
    anno_df: Optional[pd.DataFrame] = None,
) -> List[DetectronSample]:
    """Get a dataset as list of samples.

    Args:
        split: Dataset split to consider.
        base_path: Base path to dataset.
        bg_class_id: Background class index.
        ignore_bg_class: Whether to exclude background objects from labels.
            Defaults to False. ignore_bg_class should not be True when running
            evaluation because this means background objects will not have a
            ground-truth bounding box. This will wrongly incur high FPR.
        anno_df: Transform annotation. Defaults to None.

    Raises:
        ValueError: split is not among _ALLOWED_SPLITS.

    Returns:
        List of samples.
    """
    if split not in _ALLOWED_SPLITS:
        raise ValueError(
            f"split must be among {_ALLOWED_SPLITS}, but it is {split}!"
        )

    mapillary_split: Dict[str, str] = {
        "train": "training",
        "test": "validation",
        "combined": "combined",
    }[split]
    bpath: pathlib.Path = pathlib.Path(base_path)
    label_path: pathlib.Path = bpath / mapillary_split / "detectron_labels"
    img_path: pathlib.Path = bpath / mapillary_split / "images"

    dataset_dicts = []
    label_files: List[str] = [
        str(f) for f in label_path.iterdir() if f.is_file()
    ]
    label_files = sorted(label_files)

    img_df: Optional[pd.DataFrame] = None

    for idx, label_file in tqdm(enumerate(label_files)):

        filename: str = label_file.split(".txt")[0].split("/")[-1]
        jpg_filename: str = f"{filename}.jpg"
        if anno_df is not None:
            img_df = anno_df[anno_df["filename"] == jpg_filename]

        with open(label_file) as f:
            labels: List[str] = f.readlines()
            labels = [a.strip() for a in labels]

        width: float = float(labels[0].split(",")[5])
        height: float = float(labels[0].split(",")[6])
        record: DetectronSample = {
            "file_name": str(img_path / jpg_filename),
            "image_id": idx,
            "width": width,
            "height": height,
        }
        objs: List[Dict[str, Any]] = []
        for obj in labels:
            class_id, xmin, ymin, xmax, ymax, _, _, obj_id = obj.split(",")
            xmin, ymin, xmax, ymax = [
                float(x) for x in [xmin, ymin, xmax, ymax]
            ]
            class_id, obj_id = int(class_id), int(obj_id)

            if img_df is not None and not any(img_df["object_id"] == obj_id):
                # If we want results on annotated signs, we set the class of
                # the unannotated ones to "other" or background class
                class_id = bg_class_id

            # Remove "other" objects
            if ignore_bg_class and class_id == bg_class_id:
                continue

            assert class_id <= bg_class_id, (
                f"class_id {class_id} seems to be out of range ({bg_class_id} "
                "max) Something went wrong."
            )
            obj: Dict[str, Any] = {
                "bbox": [xmin, ymin, xmax, ymax],
                "bbox_mode": BoxMode.XYXY_ABS,
                "category_id": class_id,
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
    base_path: str = "~/data/",
    ignore_other: bool = False,
    class_names: List[str] = ["circle"],
    anno_df: Optional[pd.DataFrame] = None,
) -> None:
    """Register Mapillary Vistas dataset on Detectron2.

    Args:
        base_path: Base path to dataset. Defaults to "~/data/".
        ignore_other: Whether to ignore background class (last class index).
            Defaults to False.
        anno_df: Annotation DataFrame. If specified, only samples present in
            anno_df will be sampled.
        class_names: List of class names. Defaults to ["circle"].
    """
    bg_idx: int = len(class_names) - 1
    thing_classes: List[str] = class_names
    if ignore_other:
        thing_classes = thing_classes[:-1]

    splits: List[str] = ["train", "val", "combined"]
    for split in splits:
        DatasetCatalog.register(
            f"mapillary_{split}",
            lambda s=split: get_mapillary_dict(
                s,
                base_path,
                bg_idx,
                ignore_bg_class=ignore_other,
                anno_df=anno_df,
            ),
        )
        MetadataCatalog.get(f"mapillary_{split}").set(
            thing_classes=thing_classes
        )
