import json
import logging
import os
import pickle
import random

import cv2
import numpy as np
import torch
import yaml
from detectron2.data import (MetadataCatalog, build_detection_test_loader,
                             build_detection_train_loader)
from detectron2.engine import DefaultPredictor
from detectron2.evaluation import inference_on_dataset, verify_results
from detectron2.utils.visualizer import Visualizer
from tqdm import tqdm

import adv_patch_bench.utils.detectron.custom_coco_evaluator as cocoeval
from adv_patch_bench.attacks.detectron_attack_wrapper import DAGAttacker
from adv_patch_bench.dataloaders import (BenignMapper, get_mapillary_dict,
                                         get_mtsd_dict, register_mapillary,
                                         register_mtsd)
from adv_patch_bench.utils.argparse import (eval_args_parser,
                                            setup_detectron_test_args)
from adv_patch_bench.utils.detectron import build_evaluator
from hparams import (DATASETS, LABEL_LIST, NUM_CLASSES, OTHER_SIGN_CLASS, PATH_SYN_OBJ,
                     SAVE_DIR_DETECTRON, TS_NO_COLOR_LABEL_LIST)

log = logging.getLogger(__name__)
formatter = logging.Formatter('[%(levelname)s] %(asctime)s: %(message)s')


def main(cfg, args):
    # NOTE: distributed is set to False
    dataset_name = cfg.DATASETS.TEST[0]
    log.info(f'=> Creating a custom evaluator on {dataset_name}...')
    evaluator = build_evaluator(cfg, dataset_name)
    if args.debug:
        log.info(f'=> Running debug mode...')
        sampler = list(range(20))
    else:
        sampler = None
    log.info(f'=> Building {dataset_name} dataloader...')
    val_loader = build_detection_test_loader(
        cfg,
        dataset_name,
        # batch_size=cfg.SOLVER.IMS_PER_BATCH,
        batch_size=1,
        num_workers=cfg.DATALOADER.NUM_WORKERS,
        sampler=sampler,
    )
    # val_iter = iter(val_loader)
    # print(max([next(val_iter)[0]['image'].shape[0] for _ in range(5000)]))
    # import pdb
    # pdb.set_trace()
    predictor = DefaultPredictor(cfg)
    print(inference_on_dataset(predictor.model, val_loader, evaluator))


def main_single(cfg, dataset_params):
    # Build model
    model = DefaultPredictor(cfg)
    # Build dataloader
    metadata = MetadataCatalog.get(cfg.DATASETS.TEST[0])
    # val_loader = build_detection_test_loader(cfg, cfg.DATASETS.TEST[0],
    #                                          batch_size=1,
    #                                          num_workers=cfg.DATALOADER.NUM_WORKERS)
    # val_loader = build_detection_train_loader(cfg)
    split = cfg.DATASETS.TEST[0].split('_')[1]
    # val_loader = get_mtsd_dict(split, *dataset_params)
    val_loader = get_mapillary_dict(split, *dataset_params)
    for i, inpt in enumerate(val_loader):

        img = cv2.imread(inpt['file_name'])

        # DEBUG
        if args.debug:
            print(inpt['file_name'])
        if i == 10:
            break
            # import pdb
            # pdb.set_trace()
        # img = inpt[0]['image'].permute(1, 2, 0).numpy()
        # prediction = model(img)
        # visualizer = Visualizer(img[:, :, ::-1], metadata=metadata, scale=0.5)
        # out_gt = visualizer.draw_dataset_dict(inpt[0])

        # visualizer = Visualizer(img[:, :, ::-1], metadata=metadata, scale=0.5)
        # out_gt = visualizer.draw_dataset_dict(inpt)
        # out_gt.save('gt.png')
        # visualizer = Visualizer(img[:, :, ::-1], metadata=metadata, scale=0.5)
        # prediction = model(img)
        # out_pred = visualizer.draw_instance_predictions(prediction['instances'].to('cpu'))
        # out_pred.save('pred.png')


def main_attack(cfg, args, dataset_params):

    save_dir = get_save_dir(args)
    vis_dir = os.path.join(save_dir, 'vis')
    os.makedirs(vis_dir, exist_ok=True)
    args.adv_patch_path = os.path.join(save_dir, 'adv_patch.pkl')

    with open(args.attack_config_path) as file:
        attack_config = yaml.load(file, Loader=yaml.FullLoader)
        # `input_size` should be used for background size in synthetic
        # attack only
        width = cfg.INPUT.MAX_SIZE_TEST
        attack_config['input_size'] = (int(3 / 4 * width), width)

    # Build model
    model = DefaultPredictor(cfg).model

    # Build dataloader
    val_loader = build_detection_test_loader(
        cfg, cfg.DATASETS.TEST[0], mapper=BenignMapper(cfg, is_train=False),
        # cfg, cfg.DATASETS.TEST[0],
        batch_size=1, num_workers=cfg.DATALOADER.NUM_WORKERS
    )

    attack = DAGAttacker(cfg, args, attack_config, model, val_loader,
                         class_names=LABEL_LIST[args.dataset])
    if args.attack_type != 'none':
        adv_patch, patch_mask = pickle.load(open(args.adv_patch_path, 'rb'))
    else:
        patch_mask = None
    log.info('=> Running attack...')
    coco_instances_results, metrics = attack.run(
        patch_mask,
        vis_save_dir=vis_dir,
        vis_conf_thresh=0.5,
    )
    pickle.dump(metrics, open(os.path.join(save_dir, 'results.pkl'), 'wb'))

    # Logging results
    metrics = metrics['bbox']
    max_f1_idx, tp_full, fp_full, num_gts_per_class = metrics['dumped_metrics']
    metrics['dumped_metrics'] = None
    for k, v in metrics.items():
        log.info(f'{k}: {v}')
    iou_idx = 0
    tp, fp = tp_full[iou_idx, :, max_f1_idx], fp_full[iou_idx, :, max_f1_idx]
    log.info('          tp     fp     num_gt')
    for i, (t, f, n) in enumerate(zip(tp, fp, num_gts_per_class)):
        log.info(f'Class {i:2d}: {t:4d} {f:4d} {n:4d}')


def compute_metrics(cfg, args):

    dataset_name = cfg.DATASETS.TEST[0]
    print(f'=> Creating a custom evaluator on {dataset_name}...')
    evaluator = build_evaluator(cfg, dataset_name)

    # Load results from coco_instances.json
    save_dir = os.path.join(SAVE_DIR_DETECTRON, args.name)
    with open(os.path.join(save_dir, f'coco_instances_results.json')) as f:
        coco_results = json.load(f)
    img_ids = None  # Set to None to evaluate the entire dataset

    val_loader = build_detection_test_loader(
        cfg, cfg.DATASETS.TEST[0], mapper=BenignMapper(cfg, is_train=False),
        batch_size=1, num_workers=cfg.DATALOADER.NUM_WORKERS
    )
    coco_results = [[r for r in coco_results if r['image_id'] == i] for i in range(len(val_loader))]

    evaluator.reset()
    for i, batch in tqdm(enumerate(val_loader)):
        # print(batch, coco_results[i])
        # import pdb
        # pdb.set_trace()
        evaluator.process(batch, [coco_results[i]], outputs_are_json=True)
    results = evaluator.evaluate()

    # coco_eval = (
    #     cocoeval._evaluate_predictions_on_coco(
    #         evaluator._coco_api,
    #         coco_results,
    #         'bbox',
    #         kpt_oks_sigmas=evaluator._kpt_oks_sigmas,
    #         use_fast_impl=evaluator._use_fast_impl,
    #         img_ids=img_ids,
    #         eval_mode=evaluator.eval_mode,
    #         other_catId=evaluator.other_catId,
    #     )
    #     if len(coco_results) > 0
    #     else None  # cocoapi does not handle empty results very well
    # )

    # res = evaluator._derive_coco_results(
    #     coco_eval, 'bbox', class_names=evaluator._metadata.get('thing_classes')
    # )
    import pdb
    pdb.set_trace()
    print('Done')
    return


def get_save_dir(args):
    # Create folder for saving eval results
    class_names = LABEL_LIST[args.dataset]
    if args.obj_class == -1:
        class_name = 'all'
    elif 0 <= args.obj_class <= len(class_names) - 1:
        class_name = class_names[args.obj_class]
    else:
        raise ValueError((f'Invalid target object class ({args.obj_class}). '
                          f'Must be between -1 and {len(class_names) - 1}.'))
    save_dir = os.path.join(SAVE_DIR_DETECTRON, args.name, class_name)
    os.makedirs(save_dir, exist_ok=True)
    return save_dir


if __name__ == "__main__":
    args = eval_args_parser(True)
    # Verify some args
    cfg = setup_detectron_test_args(args, OTHER_SIGN_CLASS)
    assert args.dataset in DATASETS

    # Set up logger
    save_dir = get_save_dir(args)
    log.setLevel(logging.DEBUG if args.debug else logging.INFO)
    file_handler = logging.FileHandler(os.path.join(save_dir, 'results.log'),
                                       mode='a')
    file_handler.setFormatter(formatter)
    log.addHandler(file_handler)

    log.info(args)
    args.img_size = args.padded_imgsz
    # Set path to synthetic object used by synthetic attack only
    args.syn_obj_path = os.path.join(PATH_SYN_OBJ,
                                     TS_NO_COLOR_LABEL_LIST[args.obj_class])

    torch.random.manual_seed(cfg.SEED)
    np.random.seed(cfg.SEED)
    random.seed(cfg.SEED)

    # Register dataset
    if 'mtsd' in args.dataset:
        assert 'mtsd' in cfg.DATASETS.TEST[0], \
            'MTSD is specified as dataset in args but not config file'
        dataset_params = register_mtsd(
            use_mtsd_original_labels='orig' in args.dataset,
            use_color=args.use_color,
            ignore_other=args.data_no_other,
        )
    else:
        assert 'mapillary' in cfg.DATASETS.TEST[0], \
            'Mapillary is specified as dataset in args but not config file'
        dataset_params = register_mapillary(
            use_color=args.use_color,
            ignore_other=args.data_no_other,
            only_annotated=args.annotated_signs_only,
        )

    if args.compute_metrics:
        compute_metrics(cfg, args)
    elif args.single_image:
        main_single(cfg, dataset_params)
    elif args.attack_type != 'none':
        main_attack(cfg, args, dataset_params)
    else:
        # main(cfg, args)
        main_attack(cfg, args, dataset_params)
