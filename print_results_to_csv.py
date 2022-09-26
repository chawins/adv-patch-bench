import argparse
import pathlib
import pickle

import numpy as np
import pandas as pd

from hparams import (
    TS_NO_COLOR_LABEL_LIST,
    ANNO_LABEL_COUNTS_DICT,
    ANNO_NOBG_LABEL_COUNTS_DICT,
    MAPILLARY_LABEL_COUNTS_DICT,
)
# from hparams import ANNO_NOBG_LABEL_COUNTS_DICT_200 as ANNO_NOBG_LABEL_COUNTS_DICT
_NUM_IOU_THRES = 10

# Mapillary annotated
_NUM_SIGNS_PER_CLASS = np.array(list(ANNO_NOBG_LABEL_COUNTS_DICT.values()))
_NUM_SIGNS_PER_CLASS_BG = np.array(list(ANNO_LABEL_COUNTS_DICT.values()))
_BG_DIFF = _NUM_SIGNS_PER_CLASS_BG - _NUM_SIGNS_PER_CLASS

# MTSD
# _NUM_SIGNS_PER_CLASS = np.array([2999, 711, 347, 176, 1278, 287, 585, 117, 135, 30, 181])
# _BG_DIFF = np.zeros_like(_NUM_SIGNS_PER_CLASS)

# Mapillary
# _NUM_SIGNS_PER_CLASS = np.array(list(MAPILLARY_LABEL_COUNTS_DICT.values())[:-1])
# _BG_DIFF = np.zeros_like(_NUM_SIGNS_PER_CLASS)

BASE_PATH = "./detectron_output/"
CONF_THRES = 0.634
iou_idx = 0  # 0.5


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

    score_idx = np.where(scores >= CONF_THRES)[0][-1]
    
    return {
        "precision": pr[score_idx],
        "recall": rc[score_idx],
        "AP": np.mean(i_pr),
        "interpolated precision": i_pr,
        "interpolated recall": recall_thresholds,
        "total positives": NP,
        "TP": tp[-1] if len(tp) != 0 else 0,
        "FP": fp[-1] if len(fp) != 0 else 0,
    }


def _average(print_df_rows, base_sid, all_class_sid, metric_name):
    metrics = np.zeros(len(_NUM_SIGNS_PER_CLASS))
    for i in range(len(_NUM_SIGNS_PER_CLASS)):
        # if metric_name == "Precision":
        #     import pdb
        #     pdb.set_trace()
        # print(metric_name, print_df_rows[f"{base_sid}_{i:02d}"][metric_name])
        metrics[i] = print_df_rows[f"{base_sid}_{i:02d}"][metric_name]
    print_df_rows[all_class_sid][metric_name] = np.mean(metrics)
    return metrics


def main(args):
    exp_type = args.exp_type
    clean_exp_name = args.clean_exp_name
    attack_exp_name = args.attack_exp_name
    clean_exp_path = pathlib.Path(BASE_PATH) / clean_exp_name
    attack_exp_path = pathlib.Path(BASE_PATH) / attack_exp_name
    exp_paths = []
    if clean_exp_path.is_dir():
        exp_paths.extend(list(clean_exp_path.iterdir()))
    if attack_exp_path.is_dir():
        exp_paths.extend(list(attack_exp_path.iterdir()))

    df_rows = {}
    gt_scores = [{}, {}]
    results_all_classes = {}
    print_df_rows = {}
    tp_scores = {}
    fp_scores = {}
    repeated_results = []

    # Iterate over sign classes
    for sign_path in exp_paths:

        if not sign_path.is_dir():
            continue

        # Iterate over attack_type (none, load, syn_none, syn_load, etc.)
        for setting_path in sign_path.iterdir():
            result_paths = setting_path.glob("*.pkl")
            result_paths = list(result_paths)
            if not result_paths:
                continue

            # Select latest result only
            mtimes = np.array(
                [
                    float(pathlib.Path(result_path).stat().st_mtime)
                    for result_path in result_paths
                ]
            )
            latest_idx = np.argmax(mtimes)
            result_paths = [result_paths[latest_idx]]

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
                syn_rotate = int(results.get("syn_rotate_degree", 15.0))
                synthetic = int(results["synthetic"])
                is_attack = int(results["attack_type"] != "none")
                scores_dict = gt_scores[is_attack]
                # if obj_size == 64:
                #    continue

                if synthetic:
                    # Synthetic sign
                    if exp_type is not None and exp_type != "syn":
                        continue
                    cls_scores = {
                        obj_class: metrics["syn_scores"]
                        * metrics["syn_matches"]
                    }
                    syn_use_scale = int(results["syn_use_scale"])
                    syn_use_colorjitter = int(results["syn_use_colorjitter"])
                    base_sid = f"syn_size{obj_size}_rt{syn_rotate}_{syn_use_scale}_{syn_use_colorjitter}_atk{int(is_attack)}"
                else:
                    # Real signs
                    if exp_type is not None and exp_type != "real":
                        continue
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
                    obj_classes = range(len(_NUM_SIGNS_PER_CLASS))
                else:
                    obj_classes = [obj_class]

                if base_sid not in tp_scores:
                    tp_scores[base_sid] = {t: [] for t in range(10)}
                    fp_scores[base_sid] = {t: [] for t in range(10)}

                for oc in obj_classes:
                    scores = cls_scores[oc]
                    sid = f"{base_sid}_{oc:02d}"
                    if sid in scores_dict:
                        # There should only be one clean setting
                        # raise ValueError(
                        #     f"There are multiple results under same setting "
                        #     f"({sid}). Check result at {result_path}."
                        # )
                        repeated_results.append(result_path)
                        continue
                    # if not is_attack and synthetic:
                    #     import pdb
                    #     pdb.set_trace()
                    scores_dict[sid] = (time, scores)

                    tp = np.sum(scores[iou_idx] >= CONF_THRES)
                    class_name = TS_NO_COLOR_LABEL_LIST[oc]
                    tpr = tp / (
                        scores.shape[1]
                        if synthetic
                        else _NUM_SIGNS_PER_CLASS[oc]
                    )
                    metrics[f"FNR-{class_name}"] = 1 - tpr

                    print_df_rows[sid] = {
                        "id": sid,
                        "atk": is_attack,
                        "FNR": (1 - tpr) * 100,
                    }
                    if not synthetic:
                        # print_df_rows[sid]["AP"] = metrics[f"AP-{class_name}"]

                        # TODO: Recompute AP by ignoring bg signs
                        aps = np.zeros(_NUM_IOU_THRES)
                        for t in range(_NUM_IOU_THRES):
                            scores = np.concatenate(
                                [
                                    results["bbox"]["scores_full"][oc][t][0],
                                    results["bbox"]["scores_full"][oc][t][1],
                                ],
                                axis=0,
                            )
                            matches = np.zeros_like(scores, dtype=bool)
                            nm = len(results["bbox"]["scores_full"][oc][t][0])
                            matches[:nm] = 1
                            outputs = _compute_ap_recall(
                                scores, matches, _NUM_SIGNS_PER_CLASS[oc]
                            )
                            aps[t] = outputs["AP"]
                            if t == iou_idx:
                                # FIXME: precision can't be weighted average
                                print_df_rows[sid]["Precision"] = outputs["precision"] * 100
                                print_df_rows[sid]["Recall"] = outputs["recall"] * 100
                            
                        print_df_rows[sid]["AP"] = aps.mean() * 100

                        for t in range(10):
                            tp_score = results["bbox"]["scores_full"][oc][t][0]
                            tp_scores[base_sid][t].extend(tp_score)
                            fp_score = results["bbox"]["scores_full"][oc][t][1]
                            fp_scores[base_sid][t].extend(fp_score)

                # Create DF row for all classes
                all_class_sid = f"{base_sid}_all"
                print_df_rows[all_class_sid] = {
                    "id": all_class_sid,
                    "atk": is_attack,
                }
                # Weighted
                allw_class_sid = f"{base_sid}_allw"
                print_df_rows[allw_class_sid] = {
                    "id": allw_class_sid,
                    "atk": is_attack,
                }

                # Print result as one row in df
                df_row = {}
                for k, v in results.items():
                    if isinstance(v, (float, int, str, bool)):
                        df_row[k] = v
                for k, v in metrics.items():
                    if isinstance(v, (float, int, str, bool)):
                        df_row[k] = v
                df_rows[time] = df_row

    # FNR for clean syn
    fnrs = np.zeros(len(_NUM_SIGNS_PER_CLASS))
    sid_no_class = None
    for sid, data in print_df_rows.items():
        is_attack = "atk1" in sid
        if is_attack:
            continue
        base_sid = "_".join(sid.split("_")[:-1])
        all_class_sid = f"{base_sid}_all"
        allw_class_sid = f"{base_sid}_allw"
        
        if 'real' in sid:
            _average(print_df_rows, base_sid, all_class_sid, "Precision")
            _average(print_df_rows, base_sid, all_class_sid, "Recall")
            _average(print_df_rows, base_sid, all_class_sid, "AP")
        fnrs = _average(print_df_rows, base_sid, all_class_sid, "FNR")
        print_df_rows[allw_class_sid]["FNR"] = np.sum(
            fnrs
            * _NUM_SIGNS_PER_CLASS
            / np.sum(_NUM_SIGNS_PER_CLASS)
        )
        # if "syn" in sid and "atk0" in sid and "all" not in sid:
        #     sid_no_class = "_".join(sid.split("_")[:-1])
        #     k = int(sid.split("_")[-1])
        #     fnrs[k] = data["FNR"]
    # if sid_no_class is not None:
    #     print_df_rows[sid_no_class + "_all"]["FNR"] = np.mean(fnrs)
    #     print_df_rows[sid_no_class + "_allw"]["FNR"] = np.sum(
    #         fnrs * _NUM_SIGNS_PER_CLASS / np.sum(_NUM_SIGNS_PER_CLASS)
    #     )

    # Iterate through all attack experiments
    for sid, (time, adv_scores) in gt_scores[1].items():

        split_sid = sid.split("_")
        k = int(split_sid[-1])
        if "real" in split_sid:
            clean_sid = f"real_atk0_{split_sid[-1]}"
        else:
            clean_sid = "_".join([*split_sid[:-2], "atk0", split_sid[-1]])
        if clean_sid not in gt_scores[0]:
            continue
        clean_scores = gt_scores[0][clean_sid][1]
        clean_detected = clean_scores[iou_idx] >= CONF_THRES
        adv_detected = adv_scores[iou_idx] >= CONF_THRES
        total = _NUM_SIGNS_PER_CLASS[k] if "real" in split_sid else 5000

        num_succeed = np.sum(~adv_detected & clean_detected)
        num_clean = np.sum(clean_detected)
        # Account for misses caused by signs that are supposed to be in bg
        num_missed = (
            np.sum(~adv_detected) - (_BG_DIFF[k] if "real" in split_sid else 0)
        )

        attack_success_rate = num_succeed / (num_clean + 1e-9) * 100
        df_rows[time]["ASR"] = attack_success_rate
        print_df_rows[sid]["ASR"] = attack_success_rate

        sid_no_class = "_".join(split_sid[:-1])
        fnr = print_df_rows[sid]["FNR"]
        if "real" in sid_no_class:
            ap = print_df_rows[sid]["AP"]
        else:
            ap = -1e9

        # if "syn_size64_rt15_0_0_atk1_00" in sid:
        #     print(sid)
        #     print(num_succeed, num_missed, num_clean, total, attack_success_rate)
        #     import pdb
        #     pdb.set_trace()
        #     print()

        if sid_no_class in results_all_classes:
            results_all_classes[sid_no_class]["num_succeed"] += num_succeed
            results_all_classes[sid_no_class]["num_clean"][k] = num_clean
            results_all_classes[sid_no_class]["num_missed"] += num_missed
            results_all_classes[sid_no_class]["num_total"] += total
            results_all_classes[sid_no_class]["asr"][k] = attack_success_rate
            results_all_classes[sid_no_class]["fnr"][k] = fnr
            results_all_classes[sid_no_class]["ap"][k] = ap
        else:
            asrs = np.zeros(len(_NUM_SIGNS_PER_CLASS))
            asrs[k] = attack_success_rate
            fnrs = np.zeros_like(asrs)
            fnrs[k] = fnr
            aps = np.zeros_like(asrs)
            aps[k] = ap
            num_cleans = np.zeros_like(asrs)
            num_cleans[k] = num_clean
            results_all_classes[sid_no_class] = {
                "num_succeed": num_succeed,
                "num_clean": num_cleans,
                "num_missed": num_missed,
                "num_total": total,
                "asr": asrs,
                "fnr": fnrs,
                "ap": aps,
            }

    df_rows = list(df_rows.values())
    df = pd.DataFrame.from_records(df_rows)
    df = df.sort_index(axis=1)
    # df.to_csv(attack_exp_path / "results.csv")

    print(attack_exp_name, clean_exp_name, CONF_THRES)
    print("All-class ASR")
    for sid in results_all_classes:

        num_succeed = results_all_classes[sid]["num_succeed"]
        num_clean = results_all_classes[sid]["num_clean"]
        num_missed = results_all_classes[sid]["num_missed"]
        total = results_all_classes[sid]["num_total"]
        asr = num_succeed / (num_clean.sum() + 1e-9) * 100

        # Average metrics over classes instead of counting all as one
        all_class_sid = f"{sid}_all"
        asrs = results_all_classes[sid]["asr"]
        fnrs = results_all_classes[sid]["fnr"]
        avg_asr = np.mean(asrs)
        print_df_rows[all_class_sid]["ASR"] = avg_asr
        avg_fnr = np.mean(fnrs)
        print_df_rows[all_class_sid]["FNR"] = avg_fnr

        # Weighted average by number of real sign distribution
        allw_class_sid = f"{sid}_allw"
        print_df_rows[allw_class_sid]["ASR"] = np.sum(
            asrs * _NUM_SIGNS_PER_CLASS / np.sum(_NUM_SIGNS_PER_CLASS)
        )
        print_df_rows[allw_class_sid]["FNR"] = np.sum(
            fnrs * _NUM_SIGNS_PER_CLASS / np.sum(_NUM_SIGNS_PER_CLASS)
        )

        # if "syn_size64_rt15_0_0" in sid:
        #     print(sid)
        #     print(num_succeed, num_missed, num_clean, total, asr)
        #     import pdb
        #     pdb.set_trace()
        #     print(num_missed / total)

        if "real" in sid:
            # This is the correct (or commonly used) definition of mAP
            mAP = np.mean(results_all_classes[sid]["ap"])
            print_df_rows[all_class_sid]["AP"] = mAP

            aps = np.zeros(_NUM_IOU_THRES)
            num_dts = None
            for t in range(_NUM_IOU_THRES):
                matched_len = len(tp_scores[sid][t])
                unmatched_len = len(fp_scores[sid][t])
                if num_dts is not None:
                    assert num_dts == matched_len + unmatched_len
                num_dts = matched_len + unmatched_len
                scores = np.zeros(num_dts)
                matches = np.zeros_like(scores, dtype=bool)
                scores[:matched_len] = tp_scores[sid][t]
                scores[matched_len:] = fp_scores[sid][t]
                matches[:matched_len] = 1
                aps[t] = _compute_ap_recall(scores, matches, total)["AP"]
            print_df_rows[allw_class_sid]["AP"] = np.mean(aps) * 100

        print(
            f"{sid}: combined {asr:.2f} ({num_succeed}/{num_clean.sum()}), "
            f"average {avg_asr:.2f}, total {total}"
        )

    for sid in tp_scores:
        if "real" in sid and "atk0" in sid:
            aps = np.zeros(_NUM_IOU_THRES)
            num_dts = None
            for t in range(_NUM_IOU_THRES):
                matched_len = len(tp_scores[sid][t])
                unmatched_len = len(fp_scores[sid][t])
                if num_dts is not None:
                    assert num_dts == matched_len + unmatched_len
                num_dts = matched_len + unmatched_len
                scores = np.zeros(num_dts)
                matches = np.zeros_like(scores, dtype=bool)
                scores[:matched_len] = tp_scores[sid][t]
                scores[matched_len:] = fp_scores[sid][t]
                matches[:matched_len] = 1
                aps[t] = _compute_ap_recall(
                    scores, matches, _NUM_SIGNS_PER_CLASS.sum()
                )["AP"]
            print_df_rows[sid + "_allw"]["AP"] = np.mean(aps) * 100

    print_df_rows = list(print_df_rows.values())
    df = pd.DataFrame.from_records(print_df_rows)
    df = df.sort_values(["id", "atk"])
    df = df.drop(columns=["atk"])
    # df = df.reindex(columns=["id", "FNR", "ASR", "AP", "Precision", "Recall"])
    df = df.reindex(columns=["id", "FNR", "ASR", "AP"])
    print(df.to_csv(float_format="%0.2f", index=False))
    print("Repeated results:", repeated_results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("clean_exp_name", type=str, help="clean_exp_name")
    parser.add_argument("attack_exp_name", type=str, help="attack_exp_name")
    parser.add_argument(
        "--exp_type",
        type=str,
        default=None,
        required=False,
        help="real or syn (default is both)",
    )
    args = parser.parse_args()
    main(args)
