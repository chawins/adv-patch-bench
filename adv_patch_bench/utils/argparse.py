import argparse
import os
from pathlib import Path

from detectron2.config import get_cfg
from detectron2.engine import default_argument_parser, default_setup
from hparams import LABEL_LIST, NUM_CLASSES, PATH_SYN_OBJ, SAVE_DIR_DETECTRON

# _TEST_PARAMS = ['interp', 'synthetic', 'obj_size', 'syn_use_scale', 'syn_use_colorjitter']
# _ATK_PARAMS = ['attack_type', 'mask_name', 'transform_mode']


def eval_args_parser(is_detectron, root=None):

    if root is None:
        root = os.getcwd()
    root = Path(root)

    if is_detectron:
        parser = default_argument_parser()
    else:
        parser = argparse.ArgumentParser()

    parser.add_argument("--single-image", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--dataset", type=str, required=False)
    parser.add_argument(
        "--data-no-other",
        action="store_true",
        help='If True, do not load "other" or "background" class to the dataset.',
    )
    # TODO: Is this still needed? Since we can set it in hparams.
    parser.add_argument(
        "--other-class-label",
        type=int,
        default=None,
        help='Class for the "other" label.',
    )
    parser.add_argument("--eval-mode", type=str, default=None)
    parser.add_argument(
        "--interp",
        type=str,
        default="bicubic",
        help="Interpolation method: nearest, bilinear, bicubic (default).",
    )
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="evaluate with pasted synthetic signs",
    )
    parser.add_argument("--name", default="exp", help="save to project/name")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--seed", type=int, default=0, help="set random seed")
    parser.add_argument(
        "--padded-imgsz",
        type=str,
        default="3000,4000",
        help="final image size including padding (height,width). Default: 3000,4000",
    )
    parser.add_argument(
        "--annotated-signs-only",
        action="store_true",
        help="if True, only calculate metrics on annotated signs.",
    )
    parser.add_argument(
        "--weights",
        nargs="+",
        type=str,
        default=root / "yolov5s.pt",
        help="Path to PyTorch model weights.",
    )
    parser.add_argument(
        "--conf-thres",
        type=float,
        default=None,
        help=(
            "Set confidence threshold for detection."
            "Otherwise, threshold is set to max f1 score."
        ),
    )
    parser.add_argument(
        "--num-test",
        type=int,
        default=int(1e9),
        help="Max number of images to test on (default: 1e9)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Number of dataloader workers (per RANK in DDP mode).",
    )
    # syn_rotate_degree: float = 15,
    # syn_use_scale: bool = True,
    # syn_3d_transform: bool = False,
    # syn_3d_distortion: float = 0.25,
    # syn_use_colorjitter: bool = False,
    # syn_colorjitter_intensity: float = 0.3,
    # Specific to synthetic signs
    parser.add_argument(
        "--obj-size",
        type=int,
        default=None,
        help=(
            "Object width in pixels (default: 0.1 * img_size)."
            "Only used by gen_patch in for synthetic signs."
        ),
    )
    parser.add_argument(
        "--syn-rotate-degree",
        type=float,
        default=15,
        help="Max rotation degrees for synthetic sign (default: 15).",
    )
    parser.add_argument(
        "--syn-use-scale",
        action="store_true",
        help="Use scaling transform when evaluating on synthetic signs.",
    )
    parser.add_argument(
        "--syn-3d-transform",
        action="store_true",
        help="Use 3d transform when evaluating on synthetic signs.",
    )
    parser.add_argument(
        "--syn-3d-distortion",
        type=float,
        default=0.25,
        help=(
            "Perspective transform distortion for synthetic sign "
            "(default: 0.25)."
        ),
    )
    parser.add_argument(
        "--syn-use-colorjitter",
        action="store_true",
        help="Use colorjitter transform when evaluating on synthetic signs.",
    )
    parser.add_argument(
        "--syn-colorjitter-intensity",
        type=float,
        default=0.3,
        help=(
            "Color jitter intensity for brightness, contrast, saturation "
            "(default: 0.3)."
        ),
    )

    # =========================== Attack arguments ========================== #
    parser.add_argument(
        "--attack-type",
        type=str,
        default="none",
        help=(
            "Attack evaluation to run: none (default), load, per-sign, random,"
            " debug."
        ),
    )
    parser.add_argument(
        "--adv-patch-path",
        type=str,
        default=None,
        help="Path to adv patch and mask to load.",
    )
    parser.add_argument(
        "--mask-name", type=str, default="10x10", help="Specify mask shape."
    )
    # TODO: remove
    parser.add_argument(
        "--custom-patch-size",
        type=float,
        default=None,
        help="(DEPRECATED) Set custom patch size that modifies from pre-defined mask_name.",
    )
    parser.add_argument(
        "--mask-dir",
        type=str,
        default=None,
        help=(
            "Path to dir with predefined masks (default: generate a new mask)."
        ),
    )
    parser.add_argument(
        "--obj-class",
        type=int,
        default=-1,
        help="class of object to attack (-1: all classes)",
    )
    parser.add_argument(
        "--tgt-csv-filepath",
        type=str,
        required=True,
        help="path to csv which contains target points for transform",
    )
    parser.add_argument(
        "--attack-config-path",
        type=str,
        help="Path to YAML file with attack configs.",
    )
    parser.add_argument(
        "--img-txt-path",
        type=str,
        default="",
        help="path to a text file containing image filenames",
    )
    parser.add_argument(
        "--run-only-img-txt",
        action="store_true",
        help=(
            "run evaluation on images listed in img-txt-path. "
            "Otherwise, images in img-txt-path are excluded instead."
        ),
    )
    parser.add_argument(
        "--no-patch-transform",
        action="store_true",
        help=(
            "If True, do not apply patch to signs using 3D-transform. "
            "Patch will directly face camera."
        ),
    )
    parser.add_argument(
        "--transform-mode",
        type=str,
        default="perspective",
        help=(
            "transform type to use on patch during evaluation: perspective "
            "(default), affine, translate_scale. This can be different from "
            "patch generation specified in attack config."
        ),
    )
    parser.add_argument(
        "--no-patch-relight",
        action="store_true",
        help="If True, do not apply relighting transform to patch.",
    )
    parser.add_argument(
        "--min-area",
        type=float,
        default=0,
        help=(
            "Minimum area for labels. if a label has area > min_area,"
            "predictions correspoing to this target will be discarded"
        ),
    )
    # TODO: is this stil used?
    parser.add_argument(
        "--min-pred-area",
        type=float,
        default=0,
        help=(
            "Minimum area for predictions. If predicion has area < min_area and"
            " that prediction is not matched by any gt, it will be discarded."
        ),
    )
    # TODO: deprecate, set automatically given obj class
    parser.add_argument(
        "--syn-obj-path",
        type=str,
        default="",
        help="path to image of synthetic sign (used when synthetic_eval is True)",
    )
    parser.add_argument(
        "--dt-iou-thres",
        type=float,
        default=0.5,
        help=(
            "IoU threshold to consider a match between ground-truth and "
            "predicted bbox."
        ),
    )

    # ===================== Patch generation arguments ====================== #
    parser.add_argument(
        "--bg-dir", type=str, default="", help="path to background directory"
    )
    parser.add_argument(
        "--save-images", action="store_true", help="Save generated patch"
    )

    if is_detectron:
        # TODO: is this still used?
        parser.add_argument(
            "--compute-metrics",
            action="store_true",
            help="Compute metrics after running attack",
        )
    else:
        # ========================= YOLO arguments ========================== #
        parser.add_argument(
            "--data",
            type=str,
            default=root / "data/coco128.yaml",
            help="dataset.yaml path",
        )

        parser.add_argument(
            "--batch-size", type=int, default=32, help="batch size"
        )
        parser.add_argument(
            "--imgsz",
            "--img",
            "--img-size",
            type=int,
            default=640,
            help="inference size (pixels)",
        )
        parser.add_argument(
            "--iou-thres", type=float, default=0.6, help="NMS IoU threshold"
        )
        parser.add_argument(
            "--task", default="val", help="train, val, test, speed or study"
        )
        parser.add_argument(
            "--device", default="", help="cuda device, i.e. 0 or 0,1,2,3 or cpu"
        )
        parser.add_argument(
            "--single-cls",
            action="store_true",
            help="treat as single-class dataset",
        )
        parser.add_argument(
            "--augment", action="store_true", help="augmented inference"
        )
        parser.add_argument(
            "--save-txt", action="store_true", help="save results to *.txt"
        )
        parser.add_argument(
            "--save-hybrid",
            action="store_true",
            help="save label+prediction hybrid results to *.txt",
        )
        parser.add_argument(
            "--save-conf",
            action="store_true",
            help="save confidences in --save-txt labels",
        )
        parser.add_argument(
            "--save-json",
            action="store_true",
            help="save a COCO-JSON results file",
        )
        parser.add_argument(
            "--project", default=root / "runs/val", help="save to project/name"
        )
        parser.add_argument(
            "--exist-ok",
            action="store_true",
            help="existing project/name ok, do not increment",
        )
        parser.add_argument(
            "--half",
            action="store_true",
            help="use FP16 half-precision inference",
        )
        parser.add_argument(
            "--dnn",
            action="store_true",
            help="use OpenCV DNN for ONNX inference",
        )
        parser.add_argument(
            "--model-name", default="yolov5", help="yolov5 or yolor"
        )

    # ============================== Plot / log ============================= #
    parser.add_argument(
        "--save-exp-metrics",
        action="store_true",
        help="save metrics for this experiment to dataframe",
    )
    parser.add_argument(
        "--plot-single-images",
        action="store_true",
        help="save single images in a folder instead of batch images in a single plot",
    )
    parser.add_argument(
        "--plot-class-examples",
        type=str,
        default="",
        nargs="*",
        help="save single images containing individual classes in different folders.",
    )
    parser.add_argument(
        "--metrics-confidence-threshold",
        type=float,
        default=None,
        help="confidence threshold",
    )
    parser.add_argument(
        "--plot-fp",
        action="store_true",
        help="save images containing false positives",
    )

    # TODO: remove in the future
    parser.add_argument(
        "--other-class-confidence-threshold",
        type=float,
        default=0,
        help=(
            "confidence threshold at which other labels are changed if there "
            "is a match with a prediction"
        ),
    )

    args = parser.parse_args()
    verify_args(args, is_detectron)
    return args


def verify_args(args, is_detectron):
    assert args.interp in ("nearest", "bilinear", "bicubic")
    assert args.attack_type in ("none", "load", "per-sign", "random", "debug")
    if not is_detectron:
        assert args.model_name in ("yolov5", "yolor")


def parse_dataset_name(args):
    tokens = args.dataset.split("-")
    assert len(tokens) in (2, 3)
    args.dataset = (
        f"{tokens[0]}_{tokens[2]}"
        if len(tokens) == 3
        else f"{tokens[0]}_no_color"
    )
    args.use_color = "no_color" not in tokens
    # Set YOLO data yaml file
    args.data = f"{args.dataset}.yaml"
    # Set path to synthetic object used by synthetic attack only
    args.syn_obj_path = os.path.join(
        PATH_SYN_OBJ, LABEL_LIST[args.dataset][args.obj_class] + ".png"
    )
    return tokens


def get_save_dir(args):
    # Create folder for saving eval results
    class_names = LABEL_LIST[args.dataset]
    if args.obj_class == -1:
        class_name = "all"
    elif 0 <= args.obj_class <= len(class_names) - 1:
        class_name = class_names[args.obj_class]
    else:
        raise ValueError(
            (
                f"Invalid target object class ({args.obj_class}). "
                f"Must be between -1 and {len(class_names) - 1}."
            )
        )
    # test_name = ''
    # for param in _TEST_PARAMS:
    #     test_name += f'_{getattr(args, param)}'
    save_dir = os.path.join(SAVE_DIR_DETECTRON, args.name, class_name)
    os.makedirs(save_dir, exist_ok=True)
    return save_dir


def get_result_dir(args):
    result_dir = args.attack_type + ("_syn" if args.synthetic else "")
    result_dir = os.path.join(args.save_dir, result_dir)
    os.makedirs(result_dir, exist_ok=True)
    return result_dir


def setup_detectron_test_args(args, other_sign_class):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)

    # Copy dataset from args
    tokens = parse_dataset_name(args)
    if "mtsd" in tokens:
        cfg.DATASETS.TEST = ("mtsd_val",)
    else:
        cfg.DATASETS.TEST = (f"{tokens[0]}_{tokens[1]}",)

    # (Deprecated) Copy test dataset to train one since we will use
    # `build_detection_train_loader` to get labels
    # cfg.DATASETS.TRAIN = cfg.DATASETS.TEST
    cfg.SOLVER.IMS_PER_BATCH = 1
    cfg.INPUT.CROP.ENABLED = False  # Turn off augmentation for testing
    cfg.DATALOADER.NUM_WORKERS = args.workers
    cfg.eval_mode = args.eval_mode
    cfg.obj_class = args.obj_class
    cfg.other_catId = other_sign_class[args.dataset]
    cfg.conf_thres = args.conf_thres

    # Set detectron image size from argument
    cfg.INPUT.MIN_SIZE_TEST = max(
        [int(x) for x in args.padded_imgsz.split(",")]
    )
    cfg.INPUT.MAX_SIZE_TEST = cfg.INPUT.MIN_SIZE_TEST

    # Model config
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = NUM_CLASSES[args.dataset]
    cfg.OUTPUT_DIR = SAVE_DIR_DETECTRON
    weight_path = args.weights
    if isinstance(args.weights, list):
        weight_path = weight_path[0]
    assert isinstance(
        weight_path, str
    ), f"weight_path must be string, but it is {weight_path}!"
    cfg.MODEL.WEIGHTS = weight_path

    cfg.freeze()
    default_setup(cfg, args)

    # Set path to synthetic object used by synthetic attack only
    args.syn_obj_path = os.path.join(
        PATH_SYN_OBJ, LABEL_LIST[args.dataset][args.obj_class] + ".png"
    )
    args.save_dir = get_save_dir(args)
    args.result_dir = get_result_dir(args)
    return cfg


def setup_yolo_test_args(args, other_sign_class):
    parse_dataset_name(args)
    # Set to default value. This is different from conf_thres in detectron
    args.conf_thres = 0.001
