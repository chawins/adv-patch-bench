"""Test script for Detectron2 models."""

import hashlib
import json
import logging
import os
import pickle
import random
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import yaml
from detectron2 import config, data, engine

from adv_patch_bench.data.detectron import mapper, dataloader
from adv_patch_bench.evaluators.detectron_evaluator import DetectronEvaluator
from adv_patch_bench.utils.argparse import (
    eval_args_parser,
    setup_detectron_test_args,
)
from hparams import LABEL_LIST

log = logging.getLogger(__name__)
formatter = logging.Formatter("[%(levelname)s] %(asctime)s: %(message)s")

_EVAL_PARAMS = [
    "adv_patch_path",
    "attack_type",
    "conf_thres",
    "config_file",
    "dataset",
    "debug",
    "obj_class",
    "interp",
    "num_eval",
    "padded_imgsz",
    "patch_size_inch",
    "reap_transform_mode",
    "reap_use_relight",
    "seed",
    "syn_3d_dist",
    "syn_colorjitter",
    "obj_size_px",
    "syn_rotate",
    "syn_scale",
    "syn_colorjitter",
    "syn_3d_dist",
    "weights",
]


def _hash_dict(config_dict: Dict[str, Any]) -> str:
    dict_str = json.dumps(config_dict, sort_keys=True).encode("utf-8")
    # Take first 8 characters of the hash since we prefer short file name
    return hashlib.sha512(dict_str).hexdigest()[:8]


def _normalize_dict(d: Dict[str, Any], sep: str = ".") -> Dict[str, Any]:
    [flat_dict] = pd.json_normalize(d, sep=sep).to_dict(orient="records")
    flat_dict = {key: flat_dict[key] for key in sorted(flat_dict.keys())}
    return flat_dict


def _compute_metrics(
    scores_full: np.ndarray,
    num_gts_per_class: np.ndarray,
    other_sign_class: int,
    conf_thres: Optional[float] = None,
    iou_thres: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray, Optional[float]]:

    all_iou_thres = np.linspace(
        0.5, 0.95, int(np.round((0.95 - 0.5) / 0.05)) + 1, endpoint=True
    )
    iou_idx = np.where(all_iou_thres == iou_thres)[0]
    # iou_idx can be [0], and this evaluate to True
    if len(iou_idx) == 0:
        raise ValueError(f"Invalid iou_thres {iou_thres}!")
    iou_idx = int(iou_idx)

    # Find score threshold that maximizes F1 score
    eps = np.spacing(1)
    num_classes = len(scores_full)
    num_ious = len(scores_full[0])

    if conf_thres is None:
        num_scores = 1000
        scores_thres = np.linspace(0, 1, num_scores)
        tp_full = np.zeros((num_ious, num_classes, num_scores))
        fp_full = np.zeros_like(tp_full)

        for t in range(num_ious):
            for k in range(num_classes):
                for si, s in enumerate(scores_thres):
                    tp_full[t, k, si] = np.sum(
                        np.array(scores_full[k][t][0]) >= s
                    )
                    fp_full[t, k, si] = np.sum(
                        np.array(scores_full[k][t][1]) >= s
                    )

        rc = tp_full / (num_gts_per_class[None, :, None] + eps)
        pr = tp_full / (tp_full + fp_full + eps)
        f1 = 2 * pr * rc / (pr + rc + eps)
        assert np.all(f1 >= 0) and not np.any(np.isnan(f1))

        # Remove 'other' class from f1 and average over remaining classes
        f1_mean = np.delete(f1[iou_idx], other_sign_class, axis=0).mean(0)
        max_f1_idx = f1_mean.argmax()
        max_f1 = f1_mean[max_f1_idx]
        tp: np.ndarray = tp_full[iou_idx, :, max_f1_idx]
        fp: np.ndarray = fp_full[iou_idx, :, max_f1_idx]
        conf_thres = scores_thres[max_f1_idx]
        log.debug(
            f"max_f1_idx: {max_f1_idx}, max_f1: {max_f1:.4f}, conf_thres: "
            f"{conf_thres:.3f}."
        )

    else:

        log.debug(f"Using specified conf_thres of {conf_thres}...")

        tp_full = np.zeros((num_ious, num_classes))
        fp_full = np.zeros_like(tp_full)

        for t in range(num_ious):
            for k in range(num_classes):
                tp_full[t, k] = np.sum(
                    np.array(scores_full[k][t][0]) >= conf_thres
                )
                fp_full[t, k] = np.sum(
                    np.array(scores_full[k][t][1]) >= conf_thres
                )
        tp: np.ndarray = tp_full[iou_idx]
        fp: np.ndarray = fp_full[iou_idx]

    rc = tp / (num_gts_per_class + eps)
    pr = tp / (tp + fp + eps)

    # Compute combined metrics, ignoring class
    recall_cmb = tp.sum() / (num_gts_per_class.sum() + eps)

    log.debug(f"num_gts_per_class: {num_gts_per_class}")
    log.debug(f"tp: {tp}")
    log.debug(f"fp: {fp}")
    log.debug(f"precision: {pr}")
    log.debug(f"recall: {rc}")
    log.debug(f"recall_cmb: {recall_cmb}")

    return tp, fp, conf_thres


def _dump_results(
    results: Dict[str, Any],
    config_eval: Dict[str, Any],
) -> None:
    """Dump result dict to pickle file.

    Use hash of eval and attack configs for naming so only one result is saved
    per setting.

    Args:
        results: Result dict.
        config_eval: Evaluation config dict.
    """
    result_dir = config_eval["result_dir"]
    debug = config_eval["debug"]
    if debug:
        return
    # Keep only eval params that matter (uniquely identifies evaluation setting)
    cfg_eval = {}
    for param in _EVAL_PARAMS:
        cfg_eval[param] = config_eval[param]

    # Compute hash of both dicts to use as naming so we only keep one copy of
    # result in the exact same setting.
    config_eval_hash = _hash_dict(cfg_eval)
    # Attack params are already contained in name
    config_attack_hash = _hash_dict({"name": config_eval["name"]})
    result_path = os.path.join(
        result_dir,
        f"results_eval{config_eval_hash}_atk{config_attack_hash}.pkl",
    )
    with open(result_path, "wb") as f:
        pickle.dump(results, f)


def main(cfg: config.CfgNode, config: Dict[str, Dict[str, Any]]):
    """Main function.

    Args:
        cfg: Detectron config.
        config: Config dict for both eval and attack.
    """
    config_eval: Dict[str, Any] = config["eval"]
    dataset = config_eval["dataset"]
    attack_config_path = config_eval["attack_config_path"]
    class_names = LABEL_LIST[dataset]

    # Load adversarial patch and config
    if os.path.isfile(attack_config_path):
        log.info(f"Loading saved attack config from {attack_config_path}...")
        with open(attack_config_path) as f:
            # pylint: disable=unexpected-keyword-arg
            config_attack = yaml.safe_load(f, Loader=yaml.FullLoader)
    else:
        config_attack = config["attack"]

    # Build model
    model = engine.DefaultPredictor(cfg).model

    # Build dataloader
    # pylint: disable=too-many-function-args
    val_loader = data.build_detection_test_loader(
        cfg,
        cfg.DATASETS.TEST[0],
        mapper=mapper.BenignMapper(cfg, is_train=False),
        batch_size=1,
        num_workers=cfg.DATALOADER.NUM_WORKERS,
    )

    evaluator = DetectronEvaluator(
        cfg,
        config_eval,
        config_attack,
        model,
        val_loader,
        class_names=class_names,
    )
    log.info("=> Running attack...")
    _, metrics = evaluator.run()

    eval_cfg = _normalize_dict(config_eval)
    results: Dict[str, Any] = {**metrics, **eval_cfg, **config_attack}
    _dump_results(results, config_eval)

    # Logging results
    metrics: Dict[str, Any] = results["bbox"]
    conf_thres: float = config_eval["conf_thres"]

    if config_eval["synthetic"]:
        total_num_patches = metrics["total_num_patches"]
        syn_scores = metrics["syn_scores"]
        syn_matches = metrics["syn_matches"]
        all_iou_thres = metrics["all_iou_thres"]
        iou_thres = config_eval["iou_thres"]

        # Get detection for desired score and for all IoU thresholds
        detected = (syn_scores >= conf_thres) * syn_matches
        # Select desired IoU threshold
        iou_thres_idx = int(np.where(all_iou_thres == iou_thres)[0])
        tp = detected[iou_thres_idx].sum()
        fn = total_num_patches - tp
        metrics["syn_total"] = total_num_patches
        metrics["syn_tp"] = int(tp)
        metrics["syn_fn"] = int(fn)
        metrics["syn_tpr"] = tp / total_num_patches
        metrics["syn_fnr"] = fn / total_num_patches
        log.info(
            f'[Syn] Total: {metrics["syn_total"]:4d}\n'
            f'      TP: {metrics["syn_tp"]:4d} ({metrics["syn_tpr"]:.4f})\n'
            f'      FN: {metrics["syn_fn"]:4d} ({metrics["syn_fnr"]:.4f})\n'
        )
    else:
        num_gts_per_class = metrics["num_gts_per_class"]
        tp, fp, conf_thres = _compute_metrics(
            metrics["scores_full"],
            num_gts_per_class,
            config_eval["other_sign_class"],
            conf_thres,
            config_eval["iou_thres"],
        )
        if config_eval["conf_thres"] is None:
            # Update with new conf_thres
            metrics["conf_thres"] = conf_thres

        for k, v in metrics.items():
            if "syn" in k or not isinstance(v, (int, float, str, bool)):
                continue
            log.info(f"{k}: {v}")

        log.info("          tp   fp   num_gt")
        tp_all = 0
        fp_all = 0
        total = 0
        for i, (t, f, n) in enumerate(zip(tp, fp, num_gts_per_class)):
            log.info(f"Class {i:2d}: {int(t):4d} {int(f):4d} {int(n):4d}")
            metrics[f"TP-{class_names[i]}"] = t
            metrics[f"FP-{class_names[i]}"] = f
            tp_all += t
            fp_all += f
            total += n
        metrics["TPR-all"] = tp_all / total
        metrics["FPR-all"] = fp_all / total
        log.info(f'Total num patches: {metrics["total_num_patches"]}')
        _dump_results(results, config_eval, config_attack)


if __name__ == "__main__":
    config: Dict[str, Dict[str, Any]] = eval_args_parser(True)
    cfg = setup_detectron_test_args(config)
    config_eval: Dict[str, Any] = config["eval"]
    seed: int = config_eval["seed"]

    # Set up logger
    log.setLevel(logging.DEBUG if config_eval["debug"] else logging.INFO)
    file_handler = logging.FileHandler(
        os.path.join(config_eval["result_dir"], "results.log"), mode="a"
    )
    file_handler.setFormatter(formatter)
    log.addHandler(file_handler)
    log.info(config)

    # Set random seeds
    torch.random.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    class_names: List[str] = LABEL_LIST[config_eval["dataset"]]
    dataloader.setup_dataloader(config_eval, cfg, class_names)

    main(cfg, config)
