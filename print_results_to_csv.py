import argparse
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


BASE_PATH = "./detectron_output/"
# CLEAN_EXP_NAME = "no_patch"
# ATTACK_EXP_NAME = "synthetic-10x20-obj64-pd64-ld0.00001"
CONF_THRES = 0.634
iou_idx = 0  # 0.5


def main(args):
    clean_exp_name = args.clean_exp_name
    attack_exp_name = args.attack_exp_name
    # clean_exp_path = pathlib.Path(BASE_PATH) / CLEAN_EXP_NAME
    # attack_exp_path = pathlib.Path(BASE_PATH) / ATTACK_EXP_NAME
    clean_exp_path = pathlib.Path(BASE_PATH) / clean_exp_name
    attack_exp_path = pathlib.Path(BASE_PATH) / attack_exp_name
    exp_paths = list(clean_exp_path.iterdir())
    exp_paths.extend(list(attack_exp_path.iterdir()))

    df_rows = {}
    gt_scores = [{}, {}]
    results_all_classes = {}
    print_df_rows = {}

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

                if "dumped_metrics" in results["bbox"]:
                    # Skip deprecated version
                    continue
                if "obj_class" not in results:
                    continue
                if results["name"] not in (clean_exp_name, attack_exp_name):
                    continue

                # Add timestamp
                time = result_path.split("_")[-1].split(".pkl")[0]
                results["timestamp"] = time
                metrics = results["bbox"]

                # Experiment setting identifier for matching clean and attack
                obj_class = results["obj_class"]
                obj_size = results["obj_size"]
                synthetic = int(results["synthetic"])
                is_attack = int(results["attack_type"] != "none")
                scores_dict = gt_scores[is_attack]

                if synthetic:
                    # Synthetic sign
                    cls_scores = {
                        obj_class: metrics["syn_scores"]
                        * metrics["syn_matches"]
                    }
                    syn_use_scale = int(results["syn_use_scale"])
                    syn_use_colorjitter = int(results["syn_use_colorjitter"])
                    base_sid = f"syn_size{obj_size}_{syn_use_scale}_{syn_use_colorjitter}_atk{int(is_attack)}"
                else:
                    # Real signs
                    if "gtScores" not in metrics:
                        continue
                    cls_scores = metrics["gtScores"]
                    if is_attack:
                        base_sid = f"real_{results['transform_mode']}"
                        if results["no_patch_relight"]:
                            base_sid += "_nolight"
                        base_sid += "_atk1"
                    else:
                        base_sid = "real_atk0"

                if obj_class == -1:
                    obj_classes = metrics["gtScores"].keys()
                else:
                    obj_classes = [obj_class]

                for oc in obj_classes:
                    # TODO: skip other class
                    if oc == 11:
                        continue
                    scores = cls_scores[oc]
                    sid = f"{base_sid}_{oc:02d}"
                    if sid in scores_dict:
                        # There should only be one clean setting
                        raise ValueError(
                            f"There are multiple results under same setting "
                            f"({sid}). Check result at {result_path}."
                        )
                    scores_dict[sid] = (time, scores)

                    tp = np.sum(scores[iou_idx] >= CONF_THRES)
                    tpr = tp / scores.shape[1]
                    class_name = TS_NO_COLOR_LABEL_LIST[oc]
                    metrics[f"FNR-{class_name}"] = 1 - tpr

                    print_df_rows[sid] = {
                        "id": sid,
                        "atk": is_attack,
                        "FNR": (1 - tpr) * 100,
                    }
                    if not synthetic:
                        print_df_rows[sid]["AP"] = metrics[f"AP-{class_name}"]

                # Create DF row for all classes
                all_class_sid = f"{base_sid}_all"
                print_df_rows[all_class_sid] = {
                    "id": all_class_sid,
                    "atk": is_attack,
                }
                if not is_attack and not synthetic:
                    print_df_rows[all_class_sid]["FNR"] = np.mean(
                        [
                            print_df_rows[f"{base_sid}_{x:02d}"]["FNR"]
                            for x in obj_classes
                            if x != 11
                        ]
                    )
                    print_df_rows[all_class_sid]["AP"] = np.mean(
                        [
                            print_df_rows[f"{base_sid}_{x:02d}"]["AP"]
                            for x in obj_classes
                            if x != 11
                        ]
                    )

                # Print result as one row in df
                df_row = {}
                for k, v in results.items():
                    if isinstance(v, (float, int, str, bool)):
                        df_row[k] = v
                for k, v in metrics.items():
                    if isinstance(v, (float, int, str, bool)):
                        df_row[k] = v
                df_rows[time] = df_row

    # Iterate through all attack experiments
    for sid, (time, adv_scores) in gt_scores[1].items():

        split_sid = sid.split("_")
        clean_sid = "_".join([*split_sid[:-2], "atk0", split_sid[-1]])
        if clean_sid not in gt_scores[0]:
            print(clean_sid)
            print(gt_scores[0].keys())
        clean_scores = gt_scores[0][clean_sid][1]
        clean_detected = clean_scores[iou_idx] >= CONF_THRES
        adv_detected = adv_scores[iou_idx] >= CONF_THRES
        num_succeed = np.sum(~adv_detected & clean_detected)
        num_clean = np.sum(clean_detected)
        num_missed = np.sum(~adv_detected)
        attack_success_rate = num_succeed / (num_clean + 1e-9) * 100
        df_rows[time]["ASR"] = attack_success_rate
        print_df_rows[sid]["ASR"] = attack_success_rate

        sid_no_class = "_".join(split_sid[:-1])
        fnr = print_df_rows[sid]["FNR"]
        if "real" in sid_no_class:
            ap = print_df_rows[sid]["AP"]
        else:
            ap = ""

        if sid_no_class in results_all_classes:
            results_all_classes[sid_no_class]["num_succeed"] += num_succeed
            results_all_classes[sid_no_class]["num_clean"] += num_clean
            results_all_classes[sid_no_class]["num_missed"] += num_missed
            results_all_classes[sid_no_class]["num_total"] += len(adv_detected)
            results_all_classes[sid_no_class]["asr"].append(attack_success_rate)
            results_all_classes[sid_no_class]["fnr"].append(fnr)
            results_all_classes[sid_no_class]["ap"].append(ap)
        else:
            results_all_classes[sid_no_class] = {
                "num_succeed": num_succeed,
                "num_clean": num_clean,
                "num_missed": num_missed,
                "num_total": len(adv_detected),
                "asr": [attack_success_rate],
                "fnr": [fnr],
                "ap": [ap],
            }

    df_rows = list(df_rows.values())
    df = pd.DataFrame.from_records(df_rows)
    df = df.sort_index(axis=1)
    df.to_csv(attack_exp_path / "results.csv")

    print(attack_exp_name, clean_exp_name, CONF_THRES)
    print("All-class ASR")
    for sid in results_all_classes:
        num_succeed = results_all_classes[sid]["num_succeed"]
        num_clean = results_all_classes[sid]["num_clean"]
        num_missed = results_all_classes[sid]["num_missed"]
        total = results_all_classes[sid]["num_total"]
        asr = num_succeed / (num_clean + 1e-9) * 100

        # Average metrics over classes instead of counting all as one
        all_class_sid = f"{sid}_all"
        avg_asr = np.mean(results_all_classes[sid]["asr"])
        print_df_rows[all_class_sid]["ASR"] = avg_asr
        avg_fnr = np.mean(results_all_classes[sid]["fnr"])
        print_df_rows[all_class_sid]["FNR"] = avg_fnr
        if "real" in all_class_sid:
            mAP = np.mean(results_all_classes[sid]["ap"])
            print_df_rows[all_class_sid]["AP"] = mAP

        print(
            f"{sid}: combined {asr:.2f} ({num_succeed}/{num_clean}), "
            f"average {avg_asr:.2f}, total {total}"
        )

    print_df_rows = list(print_df_rows.values())
    df = pd.DataFrame.from_records(print_df_rows)
    df = df.sort_values(["id", "atk"])
    df = df.drop(columns=["atk"])
    df = df.reindex(columns=["id", "FNR", "ASR", "AP"])
    print(df.to_csv(float_format="%0.2f", index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("clean_exp_name", type=str, help="clean_exp_name")
    parser.add_argument("attack_exp_name", type=str, help="attack_exp_name")
    args = parser.parse_args()
    main(args)
