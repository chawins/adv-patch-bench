from detectron2.config import get_cfg
from detectron2.data import build_detection_test_loader
from detectron2.engine import (DefaultPredictor, default_argument_parser,
                               default_setup)
from detectron2.evaluation import inference_on_dataset, verify_results

# Import this file to register MTSD for detectron
import adv_patch_bench.dataloaders.mtsd_detectron
from adv_patch_bench.utils.custom_coco_evaluator import CustomCOCOEvaluator

DEBUG = False


def main(cfg):
    # distributed is set to False
    evaluator = CustomCOCOEvaluator('mtsd_val', cfg, False,
                                    output_dir=cfg.OUTPUT_DIR,
                                    use_fast_impl=False)
    if DEBUG:
        sampler = list(range(10))
    else:
        sampler = None
    val_loader = build_detection_test_loader(cfg, 'mtsd_val',
                                             # batch_size=cfg.SOLVER.IMS_PER_BATCH,
                                             batch_size=1,
                                             num_workers=cfg.DATALOADER.NUM_WORKERS,
                                             sampler=sampler)
    predictor = DefaultPredictor(cfg)
    print(inference_on_dataset(predictor.model, val_loader, evaluator))


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    return cfg


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    cfg = setup(args)
    main(cfg)
