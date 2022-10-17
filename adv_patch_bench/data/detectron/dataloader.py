"""Setup data loaders."""

import os
from typing import Any, Dict, List, Optional

import pandas as pd
from adv_patch_bench.data import reap_util
from adv_patch_bench.data.detectron import mapillary, mtsd, reap
from detectron2 import config


def setup_dataloader(
    config_eval: Dict[str, Any], cfg: config.CfgNode, class_names: List[str]
) -> None:
    """Register dataset for Detectron2.

    TODO(yolo): Combine with YOLO dataloader.

    Args:
        config_eval: Dictionary of eval config.
        cfg: Detectron2 config.
        class_names: List of class names.
    """
    dataset = config_eval["dataset"]
    if dataset not in cfg.DATASETS.TEST[0]:
        raise ValueError(
            f"{dataset} is specified as dataset in args but not config file!"
        )

    # Get data path
    base_data_path: str = config_eval["data_dir"]
    base_data_path = os.path.expanduser(base_data_path)
    base_data_path = os.path.join(
        base_data_path, "color" if config_eval["use_color"] else "no_color"
    )

    # Load annotation if specified
    anno_df: Optional[pd.DataFrame] = None
    if config_eval["annotated_signs_only"]:
        anno_df = reap_util.load_annotation_df(config_eval["tgt_csv_filepath"])

    if dataset in ("reap", "synthetic"):
        # Our synthetic benchmark is also based on samples in REAP
        reap.register_reap(
            base_path=base_data_path,
            synthetic=dataset == "synthetic",
            class_names=class_names,
            anno_df=anno_df,
        )
    elif "mtsd" in dataset:
        mtsd.register_mtsd(
            base_path=base_data_path,
            use_mtsd_original_labels="orig" in dataset,
            ignore_other=False,
        )
    elif "mapillary" in dataset:
        mapillary.register_mapillary(
            base_path=base_data_path,
            ignore_other=False,
            class_names=class_names,
            anno_df=anno_df,
        )
