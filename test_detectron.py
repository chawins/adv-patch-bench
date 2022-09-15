import json
import logging
import os
import pickle
import random
from collections.abc import MutableMapping
from typing import Optional

import numpy as np
import pandas as pd
import torch
import yaml
from detectron2.data import build_detection_test_loader
from detectron2.engine import DefaultPredictor
from detectron2.evaluation import inference_on_dataset

from adv_patch_bench.attacks.detectron_attack_wrapper import (
    DetectronAttackWrapper,
)
from adv_patch_bench.dataloaders import (
    BenignMapper,
    register_mapillary,
    register_mtsd,
)
from adv_patch_bench.utils.argparse import (
    eval_args_parser,
    setup_detectron_test_args,
)
from adv_patch_bench.utils.detectron import build_evaluator
from hparams import (
    DATASETS,
    LABEL_LIST,
    OTHER_SIGN_CLASS,
    TS_NO_COLOR_LABEL_LIST,
)

log = logging.getLogger(__name__)
formatter = logging.Formatter("[%(levelname)s] %(asctime)s: %(message)s")


def normalize_dict(d: MutableMapping, sep: str = ".") -> MutableMapping:
    [flat_dict] = pd.json_normalize(d, sep=sep).to_dict(orient="records")
    flat_dict = {key: flat_dict[key] for key in sorted(flat_dict.keys())}
    return flat_dict


def _compute_metrics(
    scores_full: np.ndarray,
    num_gts_per_class: np.ndarray,
    other_sign_class: int,
    conf_thres: Optional[float] = None,
    iou_thres: float = 0.5,
) -> float:

    all_iou_thres = np.linspace(
        0.5, 0.95, int(np.round((0.95 - 0.5) / 0.05)) + 1, endpoint=True
    )
    iou_idx = np.where(all_iou_thres == iou_thres)[0]
    # iou_idx can be [0], and this evaluate to True
    if len(iou_idx) == 0:
        raise ValueError(f"Invalid iou_thres {iou_thres}!")
    iou_idx = int(iou_idx)

    # Find score threshold that maximizes F1 score
    EPS = np.spacing(1)
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

        rc = tp_full / (num_gts_per_class[None, :, None] + EPS)
        pr = tp_full / (tp_full + fp_full + EPS)
        f1 = 2 * pr * rc / (pr + rc + EPS)
        assert np.all(f1 >= 0) and not np.any(np.isnan(f1))

        # Remove 'other' class from f1 and average over remaining classes
        f1_mean = np.delete(f1[iou_idx], other_sign_class, axis=0).mean(0)
        max_f1_idx = f1_mean.argmax()
        max_f1 = f1_mean[max_f1_idx]
        tp = tp_full[iou_idx, :, max_f1_idx]
        fp = fp_full[iou_idx, :, max_f1_idx]
        conf_thres = scores_thres[max_f1_idx]
        print(
            f"[DEBUG] max_f1_idx: {max_f1_idx}, max_f1: {max_f1:.4f}, "
            f"conf_thres: {conf_thres:.3f}"
        )

    else:

        print("Using specified conf_thres...")

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
        tp = tp_full[iou_idx]
        fp = fp_full[iou_idx]

    rc = tp / (num_gts_per_class + EPS)
    pr = tp / (tp + fp + EPS)

    # Compute combined metrics, ignoring class
    recall_cmb = tp.sum() / (num_gts_per_class.sum() + EPS)

    print(f"[DEBUG] num_gts_per_class: {num_gts_per_class}")
    print(f"[DEBUG] tp: {tp}")
    print(f"[DEBUG] fp: {fp}")
    print(f"[DEBUG] precision: {pr}")
    print(f"[DEBUG] recall: {rc}")
    print(f"[DEBUG] recall_cmb: {recall_cmb}")

    return tp, fp, conf_thres


def main(cfg, args):

    # NOTE: distributed is set to False
    dataset_name = cfg.DATASETS.TEST[0]
    log.info(f"=> Creating a custom evaluator on {dataset_name}...")
    evaluator = build_evaluator(cfg, dataset_name)
    if args.debug:
        log.info(f"=> Running debug mode...")
        sampler = list(range(20))
    else:
        sampler = None
    log.info(f"=> Building {dataset_name} dataloader...")
    val_loader = build_detection_test_loader(
        cfg,
        dataset_name,
        # batch_size=cfg.SOLVER.IMS_PER_BATCH,
        batch_size=1,
        num_workers=cfg.DATALOADER.NUM_WORKERS,
        sampler=sampler,
    )
    # val_iter = iter(val_loader)
    # print(max([next(val_iter)[0]['image'].shape[0] for _ in range(5000)]))
    # import pdb
    # pdb.set_trace()
    predictor = DefaultPredictor(cfg)
    print(inference_on_dataset(predictor.model, val_loader, evaluator))


def main_attack(cfg, args):

    vis_dir = os.path.join(args.result_dir, "vis")
    os.makedirs(vis_dir, exist_ok=True)

    # Load adversarial patch and config
    args.adv_patch_path = os.path.join(args.save_dir, "adv_patch.pkl")
    if os.path.isfile(args.attack_config_path):
        with open(args.attack_config_path) as file:
            attack_config = yaml.load(file, Loader=yaml.FullLoader)
            # `input_size` should be used for background size in synthetic
            # attack only
            width = cfg.INPUT.MAX_SIZE_TEST
            attack_config["input_size"] = (int(3 / 4 * width), width)
    else:
        attack_config = None

    # Build model
    model = DefaultPredictor(cfg).model

    # Build dataloader
    val_loader = build_detection_test_loader(
        cfg,
        cfg.DATASETS.TEST[0],
        mapper=BenignMapper(cfg, is_train=False),
        batch_size=1,
        num_workers=cfg.DATALOADER.NUM_WORKERS,
    )

    attack = DetectronAttackWrapper(
        cfg,
        args,
        attack_config,
        model,
        val_loader,
        class_names=LABEL_LIST[args.dataset],
    )
    log.info("=> Running attack...")
    _, metrics = attack.run(vis_save_dir=vis_dir, vis_conf_thresh=0.5)

    test_cfg = vars(args)
    test_cfg = normalize_dict(test_cfg)
    test_cfg_hash = abs(hash(json.dumps(test_cfg, sort_keys=True)))

    if attack_config is None:
        atk_cfg_hash = None
        results = {**metrics, **test_cfg}
    else:
        attack_config = normalize_dict(attack_config)
        atk_cfg_hash = abs(hash(json.dumps(attack_config, sort_keys=True)))
        results = {**metrics, **test_cfg, **attack_config}

    result_path = os.path.join(
        args.result_dir, f"results_test{test_cfg_hash}_atk{atk_cfg_hash}.pkl"
    )
    with open(result_path, "wb") as f:
        pickle.dump(results, f)

    # Logging results
    metrics = results["bbox"]
    if args.synthetic:
        log.info(
            f'[Synthetic] Total: {metrics["syn_total"]:4d}\n'
            f'            TP: {metrics["syn_tp"]:4d} ({metrics["syn_tpr"]:.4f})\n'
            f'            FN: {metrics["syn_fn"]:4d} ({metrics["syn_fnr"]:.4f})\n'
            # f'            AP: {metrics["syn_ap"]:4f}'
            # f'            AP@0.5: {metrics["syn_ap50"]:4f}'
        )
    else:

        num_gts_per_class = metrics["num_gts_per_class"]
        tp, fp, conf_thres = _compute_metrics(
            metrics["scores_full"],
            num_gts_per_class,
            OTHER_SIGN_CLASS[args.dataset],
            args.conf_thres,
            args.dt_iou_thres,
        )
        if args.conf_thres is None:
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
            metrics[f"TP-{TS_NO_COLOR_LABEL_LIST[i]}"] = t
            metrics[f"FP-{TS_NO_COLOR_LABEL_LIST[i]}"] = f
            tp_all += t
            fp_all += f
            total += n
        metrics[f"TPR-all"] = tp_all / total
        metrics[f"FPR-all"] = fp_all / total
        log.info(f'Total num patches: {metrics["total_num_patches"]}')

        with open(result_path, "wb") as f:
            pickle.dump(results, f)


if __name__ == "__main__":
    args = eval_args_parser(True)
    # Verify some args
    cfg = setup_detectron_test_args(args, OTHER_SIGN_CLASS)
    assert args.dataset in DATASETS

    # Set up logger
    log.setLevel(logging.DEBUG if args.debug else logging.INFO)
    file_handler = logging.FileHandler(
        os.path.join(args.result_dir, "results.log"), mode="a"
    )
    file_handler.setFormatter(formatter)
    log.addHandler(file_handler)

    log.info(args)
    args.img_size = args.padded_imgsz
    torch.random.manual_seed(cfg.SEED)
    np.random.seed(cfg.SEED)
    random.seed(cfg.SEED)

    # Register dataset
    if "mtsd" in args.dataset:
        assert (
            "mtsd" in cfg.DATASETS.TEST[0]
        ), "MTSD is specified as dataset in args but not config file"
        dataset_params = register_mtsd(
            use_mtsd_original_labels="orig" in args.dataset,
            use_color=args.use_color,
            ignore_other=args.data_no_other,
        )
    else:
        assert (
            "mapillary" in cfg.DATASETS.TEST[0]
        ), "Mapillary is specified as dataset in args but not config file"
        dataset_params = register_mapillary(
            use_color=args.use_color,
            ignore_other=args.data_no_other,
            only_annotated=args.annotated_signs_only,
        )

    main_attack(cfg, args)
