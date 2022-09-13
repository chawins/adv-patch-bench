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
CLEAN_EXP_NAME = "no_patch"
ATTACK_EXP_NAME = "synthetic-10x20"
CONF_THRES = 0.792
iou_idx = 0  # 0.5

clean_exp_path = pathlib.Path(BASE_PATH) / CLEAN_EXP_NAME
attack_exp_path = pathlib.Path(BASE_PATH) / ATTACK_EXP_NAME
exp_paths = list(clean_exp_path.iterdir())
exp_paths.extend(list(attack_exp_path.iterdir()))

df_rows = {}
gt_scores = [{}, {}]
results_all_classes = {}

# Iterate over sign classes
for sign_path in exp_paths:

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

            # if "bbox" not in results:
            #     import pdb

            #     pdb.set_trace()

            if "dumped_metrics" in results["bbox"]:
                # Skip deprecated version
                continue
            if "obj_class" not in results:
                continue
            conf_thres = results["conf_thres"]

            # Add timestamp
            time = result_path.split("_")[-1].split(".pkl")[0]
            results["timestamp"] = time
            metrics = results["bbox"]

            # Experiment setting identifier for matching clean and attack
            obj_class = results["obj_class"]
            obj_size = results["obj_size"]
            synthetic = int(results["synthetic"])
            syn_use_scale = int(results["syn_use_scale"])
            syn_use_colorjitter = int(results["syn_use_colorjitter"])
            if obj_class == -1 and not synthetic:
                clean_setting_id = "all_clean_real"
            else:
                clean_setting_id = f"{obj_class}_{obj_size}_{synthetic}_{syn_use_scale}_{syn_use_colorjitter}"

            is_attack = int(results["attack_type"] != "none")
            scores_dict = gt_scores[is_attack]
            if not is_attack and clean_setting_id in scores_dict:
                raise ValueError(
                    "There are multiple results on clean data. Check result at"
                    f"{result_path}."
                )

            if "syn_scores" in metrics:
                # Synthetic sign
                scores_dict[clean_setting_id] = (
                    time,
                    metrics["syn_scores"] * metrics["syn_matches"],
                )
            else:
                # Real signs
                if "gtScores" not in metrics:
                    continue
                scores_dict[clean_setting_id] = (time, metrics["gtScores"])

            # Print result as one row in df
            df_row = {}
            for k, v in results.items():
                if isinstance(v, (float, int, str, bool)):
                    df_row[k] = v
            for k, v in metrics.items():
                if isinstance(v, float):
                    df_row[k] = v
            df_rows[time] = df_row

# Iterate through all attack experiments
for clean_setting_id, (time, scores) in gt_scores[1].items():

    adv_scores = scores
    if not df_rows[time]["synthetic"]:
        obj_class = df_rows[time]["obj_class"]
        clean_scores = gt_scores[0]["all_clean_real"][1]
        clean_scores = clean_scores[obj_class]
        adv_scores = adv_scores[obj_class]
    else:
        clean_scores = gt_scores[0][clean_setting_id][1]

    clean_detected = clean_scores[iou_idx] >= CONF_THRES
    adv_detected = adv_scores[iou_idx] >= CONF_THRES
    num_succeed = np.sum(~adv_detected & clean_detected)
    num_clean = np.sum(clean_detected)
    attack_success_rate = num_succeed / (num_clean + 1e-9)
    df_rows[time]["ASR"] = attack_success_rate
    setting_id_no_class = "_".join(clean_setting_id.split("_")[1:])

    if setting_id_no_class in results_all_classes:
        results_all_classes[setting_id_no_class]["num_succeed"] += num_succeed
        results_all_classes[setting_id_no_class]["num_clean"] += num_clean
    else:
        results_all_classes[setting_id_no_class] = {
            "num_succeed": num_succeed,
            "num_clean": num_clean,
        }

df_rows = list(df_rows.values())
df = pd.DataFrame.from_records(df_rows)
df = df.sort_index(axis=1)
df.to_csv(attack_exp_path / "results.csv")

print("All-class ASR")
for sid in results_all_classes:
    asr = results_all_classes[sid]["num_succeed"] / (
        results_all_classes[sid]["num_clean"] + 1e-9
    )
    print(f"{sid}: {asr:.4f}")
