from typing import Any, Dict, List, Tuple

from adv_patch_bench.dataloaders.detectron import mapillary
from detectron2.data import DatasetCatalog, MetadataCatalog
from hparams import TS_NO_COLOR_LABEL_LIST


def get_reap_dict(
    base_path: str,
    bg_idx: int,
) -> List[Dict[str, Any]]:

    data_dict = mapillary.get_mapillary_dict(
        split="combined",
        base_path=base_path,
        bg_idx=bg_idx,
        use_color=False,
        ignore_other=False,
        only_annotated=True,
    )
    return data_dict


def register_reap(
    base_path: str = "./data/",
    synthetic: bool = False,
) -> Tuple[Any, ...]:

    dataset_name = "synthetic" if synthetic else "reap"
    thing_classes = TS_NO_COLOR_LABEL_LIST
    # Get index of background or "other" class
    bg_idx = len(thing_classes) - 1

    DatasetCatalog.register(
        dataset_name,
        lambda: get_reap_dict(
            base_path,
            bg_idx,
        ),
    )
    MetadataCatalog.get(dataset_name).set(thing_classes=thing_classes)

    return base_path, bg_idx, False, False, True
