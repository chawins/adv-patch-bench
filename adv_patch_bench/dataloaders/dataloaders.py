import os
from typing import Any, Dict

from adv_patch_bench.dataloaders.detectron import mapillary, mtsd, reap
from detectron2 import config
from hparams import DEFAULT_DATA_PATHS


def setup_dataloader(config_eval: Dict[str, Any], cfg: config.CfgNode):
    """Register dataset."""

    # TODO(yolo): Combine with YOLO dataloader
    dataset = config_eval["dataset"]
    use_color = config_eval["use_color"]

    base_data_path = config_eval["data_dir"]
    if base_data_path is None:
        base_data_path = DEFAULT_DATA_PATHS[dataset.split("-")[0]]
    base_data_path = os.path.expanduser(base_data_path)

    if dataset not in cfg.DATASETS.TEST[0]:
        raise ValueError(
            f"{dataset} is specified as dataset in args but not config file!"
        )

    if dataset in ("reap", "synthetic"):
        # Our synthetic benchmark is also based on samples in REAP
        dataset_params = reap.register_reap(
            base_path=base_data_path,
            synthetic=dataset == "synthetic",
        )
    elif "mtsd" in dataset:
        dataset_params = mtsd.register_mtsd(
            base_path=base_data_path,
            use_mtsd_original_labels="orig" in dataset,
            use_color=use_color,
            ignore_other=False,
        )
    elif "mapillary" in dataset:
        dataset_params = mapillary.register_mapillary(
            base_path=base_data_path,
            use_color=use_color,
            ignore_other=False,
            only_annotated=config_eval["annotated_signs_only"],
        )

    return dataset_params
