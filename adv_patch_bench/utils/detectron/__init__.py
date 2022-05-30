import adv_patch_bench.utils.detectron.custom_coco_evaluator as cocoeval

from .custom_sampler import (RepeatFactorTrainingSampler,
                             ShuffleInferenceSampler)


def build_evaluator(cfg, dataset_name, output_folder=None, is_test=False):
    evaluator = cocoeval.CustomCOCOEvaluator(
        dataset_name, cfg, False,
        output_dir=cfg.OUTPUT_DIR if not is_test else None,
        use_fast_impl=False,  # Use COCO original eval code
    )
    # return COCOEvaluator('mtsd_val', output_dir=cfg.OUTPUT_DIR)
    return evaluator
