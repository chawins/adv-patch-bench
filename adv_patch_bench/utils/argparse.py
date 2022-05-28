import argparse

from detectron2.config import get_cfg
from detectron2.engine import default_argument_parser, default_setup


def eval_args_parser(is_detectron, root=None):
    if is_detectron:
        parser = default_argument_parser()
    else:
        parser = argparse.ArgumentParser()

    parser.add_argument('--single-image', action='store_true')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--dataset', type=str, required=False)
    parser.add_argument('--data-no-other', action='store_true',
                        help='If True, do not load "other" or "background" class to the dataset.')
    parser.add_argument('--other-class-label', type=int, default=None, help="Class for the 'other' label")
    parser.add_argument('--eval-mode', type=str, default=None)

    parser.add_argument('--interp', type=str, default=None,
                        help='interpolation method (nearest, bilinear, bicubic)')
    parser.add_argument('--synthetic', action='store_true',
                        help='evaluate with pasted synthetic signs')
    parser.add_argument('--name', default='exp', help='save to project/name')
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--seed', type=int, default=0, help='set random seed')
    parser.add_argument('--padded-imgsz', type=str, default='3000,4000',
                        help='final image size including padding (height,width). Default: 3000,4000')

    # =========================== Attack arguments ========================== #
    parser.add_argument('--attack-type', type=str, default='none',
                        help='which attack evaluation to run (none, load, per-sign, random, debug)')
    parser.add_argument('--adv-patch-path', type=str, default=None,
                        help='path to adv patch and mask to load')
    parser.add_argument('--mask-dir', type=str, default='./masks/',
                        help='Path to dir with predefined masks')
    parser.add_argument('--obj-class', type=int, default=-1,
                        help='class of object to attack (-1: all classes)')
    parser.add_argument('--tgt-csv-filepath', required=True,
                        help='path to csv which contains target points for transform')
    parser.add_argument('--attack-config-path',
                        help='path to yaml file with attack configs (used when attack_type is per-sign)')
    parser.add_argument('--syn-obj-path', type=str, default='',
                        help='path to an image of a synthetic sign (used when synthetic_eval is True')
    parser.add_argument('--img-txt-path', type=str, default='',
                        help='path to a text file containing image filenames')
    parser.add_argument('--run-only-img-txt', action='store_true',
                        help='run evaluation on images listed in img-txt-path. Otherwise, exclude these images.')
    parser.add_argument('--no-patch-transform', action='store_true',
                        help=('If True, do not apply patch to signs using '
                              '3D-transform. Patch will directly face camera.'))
    parser.add_argument('--no-patch-relight', action='store_true',
                        help=('If True, do not apply relighting transform to patch'))
    parser.add_argument('--min-area', type=float, default=0,
                        help=('Minimum area for labels. if a label has area > min_area,'
                              'predictions correspoing to this target will be discarded'))
    parser.add_argument('--min-pred-area', type=float, default=0,
                        help=('Minimum area for predictions. if a predicion has area < min_area and '
                              'that prediction is not matched to any label, it will be discarded'))

    # ===================== Patch generation arguments ====================== #
    parser.add_argument('--obj-size', type=int, default=None,
                        help='Object width in pixels (default: 0.1 * img_size)')
    parser.add_argument('--patch-size-inch', type=int, default=None,
                        help='Patch size in inches (deprecated)')
    parser.add_argument('--bg-dir', type=str, default='',
                        help='path to background directory')
    parser.add_argument('--num-bg', type=float, default=1,
                        help='Number of backgrounds used to generate patch')
    parser.add_argument('--save-images', action='store_true',
                        help='Save generated patch')
    # parser.add_argument('--detectron', action='store_true', help='Model is detectron else YOLO')

    if is_detectron:
        parser.add_argument('--compute-metrics', action='store_true',
                            help='Compute metrics after running attack')
    else:
        # ========================= YOLO arguments ========================== #
        parser.add_argument('--data', type=str, default=root / 'data/coco128.yaml', help='dataset.yaml path')
        parser.add_argument('--weights', nargs='+', type=str, default=root / 'yolov5s.pt', help='model.pt path(s)')
        parser.add_argument('--batch-size', type=int, default=32, help='batch size')
        parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=640, help='inference size (pixels)')
        parser.add_argument('--conf-thres', type=float, default=0.001, help='confidence threshold')
        parser.add_argument('--iou-thres', type=float, default=0.6, help='NMS IoU threshold')
        parser.add_argument('--task', default='val', help='train, val, test, speed or study')
        parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
        parser.add_argument('--workers', type=int, default=8, help='max dataloader workers (per RANK in DDP mode)')
        parser.add_argument('--single-cls', action='store_true', help='treat as single-class dataset')
        parser.add_argument('--augment', action='store_true', help='augmented inference')
        parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
        parser.add_argument('--save-hybrid', action='store_true', help='save label+prediction hybrid results to *.txt')
        parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
        parser.add_argument('--save-json', action='store_true', help='save a COCO-JSON results file')
        parser.add_argument('--project', default=root / 'runs/val', help='save to project/name')
        parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
        parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
        parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
        parser.add_argument('--model-name', default='yolov5', help='yolov5 or yolor')
        parser.add_argument('--annotated-signs-only', action='store_true', help='if True, only calculate metrics on annotated signs')

    # ============================== Plot / log ============================= #
    parser.add_argument('--save-exp-metrics', action='store_true', help='save metrics for this experiment to dataframe')
    parser.add_argument('--plot-single-images', action='store_true',
                        help='save single images in a folder instead of batch images in a single plot')
    parser.add_argument('--plot-class-examples', type=str, default='', nargs='*',
                        help='save single images containing individual classes in different folders.')
    parser.add_argument('--metrics-confidence-threshold', type=float, default=None, help='confidence threshold')
    parser.add_argument('--plot-fp', action='store_true', help='save images containing false positives')

    # TODO: remove in the future
    parser.add_argument(
        '--other-class-confidence-threshold', type=float, default=0,
        help='confidence threshold at which other labels are changed if there is a match with a prediction')

    return parser.parse_args()


def parse_dataset_name(args):
    tokens = args.dataset.split('-')
    assert len(tokens) in (2, 3)
    args.dataset = f'{tokens[0]}_{tokens[2]}' if len(tokens) == 3 else f'{tokens[0]}_no_color'
    args.use_color = 'no_color' not in tokens
    # Set YOLO data yaml file
    args.data = f'{args.dataset}.yaml'
    return tokens


def setup_detectron_test_args(args, other_sign_class):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)

    # Copy dataset from args
    tokens = parse_dataset_name(args)
    cfg.DATASETS.TEST = (f'{tokens[0]}_{tokens[1]}', )

    # Copy test dataset to train one since we will use
    # `build_detection_train_loader` to get labels
    cfg.DATASETS.TRAIN = cfg.DATASETS.TEST
    cfg.SOLVER.IMS_PER_BATCH = 1
    cfg.INPUT.CROP.ENABLED = False
    cfg.eval_mode = args.eval_mode
    cfg.other_catId = other_sign_class[args.dataset]

    # Set detectron image size from argument
    cfg.INPUT.MIN_SIZE_TEST = max([int(x) for x in args.padded_imgsz.split(',')])
    cfg.INPUT.MAX_SIZE_TEST = cfg.INPUT.MIN_SIZE_TEST

    cfg.freeze()
    default_setup(cfg, args)
    return cfg
