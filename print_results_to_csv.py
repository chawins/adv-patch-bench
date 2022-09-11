import pathlib
import pickle

import numpy as np
import pandas as pd

from hparams import TS_NO_COLOR_LABEL_LIST


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


#
# syn_use_scale
# syn_use_colorjitter
# obj_class
# obj_size
# time


BASE_PATH = "./detectron_output/"
# EXP_NAME = "synthetic-10x20"
EXP_NAME = "no_patch"
iou_idx = 0

exp_path = pathlib.Path(BASE_PATH) / EXP_NAME

df_rows = []

# Iterate over sign classes
for sign_path in exp_path.iterdir():

    if not sign_path.is_dir():
        continue

    # Iterate over attack_type (none, load, syn_none, syn_load, etc.)
    for setting_path in sign_path.iterdir():
        result_paths = setting_path.glob("*.pkl")

        # Iterate over result pickle files
        for result_path in result_paths:

            result_path = str(result_path)
            with open(result_path, "rb") as f:
                results = pickle.load(f)

            if "obj_class" not in results:
                continue
            conf_thres = results["conf_thres"]

            # Add timestamp
            time = result_path.split("_")[-1].split(".pkl")[0]
            results["timestamp"] = time

            if "dumped_metrics" in results["bbox"]:
                # Real sign
                metrics = results["bbox"]["dumped_metrics"]
                max_f1_idx, tp_full, fp_full, num_gts_per_class = metrics
                tp = tp_full[iou_idx, :, max_f1_idx]
                fp = fp_full[iou_idx, :, max_f1_idx]
                total = results["bbox"]["total_num_patches"]

                for i, (t, f, n) in enumerate(zip(tp, fp, num_gts_per_class)):
                    if i != results["obj_class"] and results["obj_class"] != -1:
                        continue
                    class_name = TS_NO_COLOR_LABEL_LIST[i]
                    df_row[f"TP-{class_name}"] = int(t)
                    df_row[f"FP-{class_name}"] = int(f)
                    df_row[f"Total-{class_name}"] = int(n)
                    df_row[f"TPR-{class_name}"] = t / n if n > 0 else 0
                    df_row[f"FNR-{class_name}"] = 1 - (t / n) if n > 0 else 0

                # TODO
            else:
                # Synthetic sign
                # syn_scores = results["syn_scores"]
                # syn_matches = results["syn_matches"]
                # # Pick IoU 0.5 which is at index 0
                # detected = ((syn_scores >= conf_thres) * syn_matches)[iou_idx]
                df_row = {}

            # Print result as one row in df
            for k, v in results.items():
                if isinstance(v, (float, int, str, bool)):
                    df_row[k] = v
            for k, v in results["bbox"].items():
                if isinstance(v, float):
                    df_row[k] = v
            df_rows.append(df_row)

df = pd.DataFrame.from_records(df_rows)
df = df.sort_index(axis=1)
df.to_csv(exp_path / "results.csv")
