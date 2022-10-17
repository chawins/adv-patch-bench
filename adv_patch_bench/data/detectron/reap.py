"""Register and load REAP benchmark as well as its synthetic version."""

from typing import List, Optional

import pandas as pd
from adv_patch_bench.data.detectron import mapillary
from adv_patch_bench.utils.types import DetectronSample
from detectron2.data import DatasetCatalog, MetadataCatalog


def get_reap_dict(
    base_path: str, bg_class_id: int, anno_df: Optional[pd.DataFrame] = None
) -> List[DetectronSample]:
    """Load REAP dataset through Mapillary Vistas loader.

    See mapillary.get_mapillary_dict() for args and returns.
    """
    data_dict = mapillary.get_mapillary_dict(
        split="combined",
        base_path=base_path,
        bg_class_id=bg_class_id,
        ignore_bg_class=False,
        anno_df=anno_df,
    )
    return data_dict


def register_reap(
    base_path: str = "./data/",
    synthetic: bool = False,
    class_names: List[str] = ["circle"],
    anno_df: Optional[pd.DataFrame] = None,
) -> None:
    """Register REAP dataset in Detectron2.

    Args:
        base_path: Base path to dataset. Defaults to "./data/".
        synthetic: Whether to use synthetic version. Defaults to False.
        class_names: List of class names. Defaults to ["circle"].
        anno_df: Annotation DataFrame. If specified, only samples present in
            anno_df will be sampled.
    """
    dataset_name: str = "synthetic" if synthetic else "reap"
    # Get index of background or "other" class
    bg_class_id: int = len(class_names) - 1

    DatasetCatalog.register(
        dataset_name,
        lambda: get_reap_dict(
            base_path,
            bg_class_id,
            anno_df,
        ),
    )
    MetadataCatalog.get(dataset_name).set(thing_classes=class_names)
