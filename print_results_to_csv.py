import pathlib
import pickle

import numpy as np
import pandas as pd


def _compute_ap_recall(scores, matched, NP, recall_thresholds=None):
    """
    (DEPRECATED) This curve tracing method has some quirks that do not appear
    when only unique confidence thresholds are used (i.e. Scikit-learn's
    implementation), however, in order to be consistent, the COCO's method is
    reproduced.
    """

    # by default evaluate on 101 recall levels
    if recall_thresholds is None:
        recall_thresholds = np.linspace(
            0.0, 1.00, int(np.round((1.00 - 0.0) / 0.01)) + 1, endpoint=True
        )

    # sort in descending score order
    inds = np.argsort(-scores, kind="stable")

    scores = scores[inds]
    matched = matched[inds]

    tp = np.cumsum(matched)
    fp = np.cumsum(~matched)

    rc = tp / NP
    pr = tp / (tp + fp)

    # make precision monotonically decreasing
    i_pr = np.maximum.accumulate(pr[::-1])[::-1]

    rec_idx = np.searchsorted(rc, recall_thresholds, side="left")

    # get interpolated precision values at the evaluation thresholds
    i_pr = np.array([i_pr[r] if r < len(i_pr) else 0 for r in rec_idx])

    return {
        "precision": pr,
        "recall": rc,
        "AP": np.mean(i_pr),
        "interpolated precision": i_pr,
        "interpolated recall": recall_thresholds,
        "total positives": NP,
        "TP": tp[-1] if len(tp) != 0 else 0,
        "FP": fp[-1] if len(fp) != 0 else 0,
    }


BASE_PATH = "./detectron_output/"
# EXP_NAME = "synthetic-10x20"
EXP_NAME = "no_patch"

exp_path = pathlib.Path(BASE_PATH) / EXP_NAME

df_rows = []

for sign_path in exp_path.iterdir():

    if not sign_path.is_dir():
        continue

    for setting_path in sign_path.iterdir():
        result_paths = setting_path.glob("*.pkl")

        for result_path in result_paths:

            # TODO: add time

            with open(result_path, "rb") as f:
                results = pickle.load(f)
            if "obj_class" not in results:
                continue

            if "dumped_metrics" in results["bbox"]:
                metrics = results["bbox"]["dumped_metrics"]
                max_f1_idx, tp_full, fp_full, num_gts_per_class = metrics
                iou_idx = 0
                tp = tp_full[iou_idx, :, max_f1_idx]
                fp = fp_full[iou_idx, :, max_f1_idx]

                df_row = {
                    "tp": tp,
                    "fp": fp,
                    "total": results["bbox"]["total_num_patches"],
                }
            else:
                df_row = {}

            for k, v in results.items():
                if isinstance(v, (float, int, str, bool)):
                    df_row[k] = v
            for k, v in results["bbox"].items():
                if isinstance(v, float):
                    df_row[k] = v
            df_rows.append(df_row)

df = pd.DataFrame.from_records(df_rows)
df.to_csv(exp_path / "results.csv")
