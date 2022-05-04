#!/usr/bin/env python
"""
A main training script.
This scripts reads a given config file and runs the training or evaluation.
It is an entry point that is made to train standard models in detectron2.
In order to let one script support training of many models,
this script contains logic that are specific to these built-in models and therefore
may not be suitable for your own project.
For example, your research project perhaps only needs a single "evaluator".
Therefore, we recommend you to use detectron2 as an library and take
this file as an example of how to use the library.
You may want to write your own script with your datasets and other customizations.
"""
import detectron2.utils.comm as comm
import torch.multiprocessing
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import (build_detection_train_loader,
                             get_detection_dataset_dicts)
from detectron2.engine import (DefaultTrainer, default_argument_parser,
                               default_setup, hooks, launch)
from detectron2.evaluation import COCOEvaluator, verify_results
from detectron2.modeling import GeneralizedRCNNWithTTA

import adv_patch_bench.utils.detectron.custom_coco_evaluator as cocoeval
# Import this file to register MTSD for detectron
from adv_patch_bench.dataloaders.detectron.mtsd import register_mtsd
from adv_patch_bench.utils.detectron.custom_best_checkpointer import \
    BestCheckpointer
from adv_patch_bench.utils.detectron.custom_sampler import \
    RepeatFactorTrainingSampler
from hparams import DATASETS, OTHER_SIGN_CLASS

torch.multiprocessing.set_sharing_strategy('file_system')


def build_evaluator(cfg, dataset_name, output_folder=None):
    evaluator = cocoeval.CustomCOCOEvaluator(
        dataset_name, cfg, False,
        output_dir=cfg.OUTPUT_DIR,
        use_fast_impl=False,  # Use COCO original eval code
        eval_mode=cfg.eval_mode,
        other_catId=cfg.other_catId,
    )
    # return COCOEvaluator('mtsd_val', output_dir=cfg.OUTPUT_DIR)
    return evaluator


class Trainer(DefaultTrainer):
    """
    We use the "DefaultTrainer" which contains pre-defined default logic for
    standard training workflow. They may not work for you, especially if you
    are working on a new research project. In that case you can write your
    own training loop. You can use "tools/plain_train_net.py" as an example.
    """

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        return build_evaluator(cfg, dataset_name, output_folder)

    # @classmethod
    # def test_with_TTA(cls, cfg, model):
    #     logger = logging.getLogger("detectron2.trainer")
    #     # In the end of training, run an evaluation with TTA
    #     # Only support some R-CNN models.
    #     logger.info("Running inference with test-time augmentation ...")
    #     model = GeneralizedRCNNWithTTA(cfg, model)
    #     evaluators = [
    #         cls.build_evaluator(
    #             cfg, name, output_folder=os.path.join(cfg.OUTPUT_DIR, "inference_TTA")
    #         )
    #         for name in cfg.DATASETS.TEST
    #     ]
    #     res = cls.test(cfg, model, evaluators)
    #     res = OrderedDict({k + "_TTA": v for k, v in res.items()})
    #     return res

    @classmethod
    def build_train_loader(cls, cfg):
        """
        Returns:
            iterable

        It now calls :func:`detectron2.data.build_detection_train_loader`.
        Overwrite it if you'd like a different data loader.
        """
        dataset = get_detection_dataset_dicts(
            cfg.DATASETS.TRAIN,
            filter_empty=cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS,
            min_keypoints=cfg.MODEL.ROI_KEYPOINT_HEAD.MIN_KEYPOINTS_PER_IMAGE
            if cfg.MODEL.KEYPOINT_ON
            else 0,
            proposal_files=cfg.DATASETS.PROPOSAL_FILES_TRAIN if cfg.MODEL.LOAD_PROPOSALS else None,
        )
        repeat_factors = RepeatFactorTrainingSampler.repeat_factors_from_category_frequency(
            dataset, cfg.DATALOADER.REPEAT_THRESHOLD
        )
        return build_detection_train_loader(cfg, sampler=RepeatFactorTrainingSampler(repeat_factors))


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)

    # Set some custom cfg from args
    cfg.eval_mode = args.eval_mode
    cfg.other_catId = OTHER_SIGN_CLASS[args.dataset]

    cfg.freeze()
    default_setup(cfg, args)
    return cfg


def main(args):

    register_mtsd(
        use_mtsd_original_labels='orig' in args.dataset,
        use_color='no_color' not in args.dataset,
        ignore_other=args.data_no_other,
    )

    cfg = setup(args)

    if args.eval_only:
        model = Trainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        res = Trainer.test(cfg, model)
        if cfg.TEST.AUG.ENABLED:
            res.update(Trainer.test_with_TTA(cfg, model))
        if comm.is_main_process():
            verify_results(cfg, res)
        return res

    """
    If you'd like to do anything fancier than the standard training logic,
    consider writing your own training loop (see plain_train_net.py) or
    subclassing the trainer.
    """
    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    # Register hook for our custom checkpointer every `eval_period` steps
    # trainer.register_hooks([
    #     BestCheckpointer(cfg.TEST.EVAL_PERIOD, trainer.checkpointer),
    # ])
    # if cfg.TEST.AUG.ENABLED:
    #     trainer.register_hooks(
    #         [hooks.EvalHook(0, lambda: trainer.test_with_TTA(cfg, trainer.model))]
    #     )
    return trainer.train()


if __name__ == "__main__":
    parser = default_argument_parser()
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--data-no-other', action='store_true',
                        help='If True, do not load "other" or "background" class to the dataset.')
    parser.add_argument('--eval-mode', type=str, default='default')
    args = parser.parse_args()

    print('Command Line Args: ', args)
    assert args.dataset in DATASETS

    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
