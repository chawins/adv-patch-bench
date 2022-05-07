import os
import pickle

import cv2
import yaml
from detectron2.config import get_cfg
from detectron2.data import (MetadataCatalog, build_detection_test_loader,
                             build_detection_train_loader)
from detectron2.engine import (DefaultPredictor, default_argument_parser,
                               default_setup)
from detectron2.evaluation import inference_on_dataset, verify_results
from detectron2.utils.visualizer import Visualizer

import adv_patch_bench.utils.detectron.custom_coco_evaluator as cocoeval
from adv_patch_bench.attacks.detectron_attack_wrapper import DAGAttacker
from adv_patch_bench.dataloaders import (BenignMapper, get_mtsd_dict,
                                         register_mapillary, register_mtsd)
from adv_patch_bench.utils.argparse import eval_args_parser
from hparams import DATASETS, LABEL_LIST, OTHER_SIGN_CLASS


def main(cfg, args):
    # NOTE: distributed is set to False
    dataset_name = args.dataset.split('_')[0]
    dataset_name = f'{dataset_name}_val'
    print(f'=> Creating a custom evaluator on {dataset_name}...')
    evaluator = cocoeval.CustomCOCOEvaluator(
        dataset_name, cfg, False,
        output_dir=cfg.OUTPUT_DIR,
        use_fast_impl=False,  # Use COCO original eval code
        eval_mode=args.eval_mode,
        other_catId=OTHER_SIGN_CLASS[args.dataset],
    )
    if args.debug:
        print(f'=> Running debug mode...')
        sampler = list(range(10))
    else:
        sampler = None
    print(f'=> Building {dataset_name} dataloader...')
    val_loader = build_detection_test_loader(cfg, dataset_name,
                                             # batch_size=cfg.SOLVER.IMS_PER_BATCH,
                                             batch_size=1,
                                             num_workers=cfg.DATALOADER.NUM_WORKERS,
                                             sampler=sampler)
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
    val_loader = get_mtsd_dict('val', *dataset_params)
    for i, inpt in enumerate(val_loader):

        img = cv2.imread(inpt['file_name'])

        # DEBUG
        if args.debug:
            import pdb
            pdb.set_trace()
        # img = inpt[0]['image'].permute(1, 2, 0).numpy()
        # prediction = model(img)
        # visualizer = Visualizer(img[:, :, ::-1], metadata=metadata, scale=0.5)
        # out_gt = visualizer.draw_dataset_dict(inpt[0])

        visualizer = Visualizer(img[:, :, ::-1], metadata=metadata, scale=0.5)
        out_gt = visualizer.draw_dataset_dict(inpt)
        out_gt.save('gt.png')
        visualizer = Visualizer(img[:, :, ::-1], metadata=metadata, scale=0.5)
        prediction = model(img)
        out_pred = visualizer.draw_instance_predictions(prediction['instances'].to('cpu'))
        out_pred.save('pred.png')


def main_attack(cfg, args, dataset_params):

    # Create folder for saving eval results
    save_dir = os.path.join('./detectron_output/', args.name)
    os.makedirs(save_dir, exist_ok=True)

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
        batch_size=1, num_workers=cfg.DATALOADER.NUM_WORKERS
    )
    # TODO: To generate more dense proposals
    # cfg.MODEL.RPN.NMS_THRESH = nms_thresh
    # cfg.MODEL.RPN.POST_NMS_TOPK_TEST = 5000
    attack = DAGAttacker(cfg, args, attack_config, model, val_loader,
                         class_names=LABEL_LIST[args.dataset])
    adv_patch, patch_mask = pickle.load(open(args.adv_patch_path, 'rb'))
    out = attack.run(
        args.obj_class,
        patch_mask,
        results_save_path=os.path.join(save_dir, f'coco_instances_results_{args.name}.json'),
        vis_save_dir=save_dir,
        vis_conf_thresh=0.5,  # TODO
    )

    # val_loader = build_detection_train_loader(cfg)
    # val_loader = get_mtsd_dict('val', *dataset_params)
    # for i, inpt in enumerate(val_loader):

    #     inpt = inpt[0]
    #     img = inpt['image'].permute(1, 2, 0)

    #     # DEBUG
    #     if args.debug:
    #         print(inpt['file_name'], img.shape)
    #         import pdb
    #         pdb.set_trace()

    #     # Visualize ground truth data
    #     if i < 10:
    #         visualizer = Visualizer(img.numpy()[:, :, ::-1], metadata=metadata, scale=0.5)
    #         out_gt = visualizer.draw_dataset_dict(inpt[0])
    #         out_gt.save(f'detectron_gt_{i + 1}.png')

    #     import pdb
    #     pdb.set_trace()

    #     prediction = model(img)

    #     # Visualize adversarial examples
    #     if i < 10:
    #         visualizer = Visualizer(img[:, :, ::-1], metadata=metadata, scale=0.5)
    #         out_pred = visualizer.draw_instance_predictions(prediction['instances'].to('cpu'))
    #         out_pred.save(f'detectron_adv_pred_{i + 1}.png')


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    # Copy test dataset to train one since we will use
    # `build_detection_train_loader` to get labels
    cfg.DATASETS.TRAIN = cfg.DATASETS.TEST
    cfg.SOLVER.IMS_PER_BATCH = 1
    cfg.INPUT.CROP.ENABLED = False
    cfg.freeze()
    default_setup(cfg, args)
    return cfg


if __name__ == "__main__":
    args = eval_args_parser(True)
    print('Command Line Args:', args)
    args.img_size = args.padded_imgsz

    # Verify some args
    assert args.dataset in DATASETS
    cfg = setup(args)

    # Register dataset
    if 'mtsd' in args.dataset:
        assert 'mtsd' in cfg.DATASETS.TEST[0], 'MTSD is specified as dataset in args but not config file'
        dataset_params = register_mtsd(
            use_mtsd_original_labels='orig' in args.dataset,
            use_color='no_color' not in args.dataset,
            ignore_other=args.data_no_other)
    else:
        assert 'mapillary' in cfg.DATASETS.TEST[0], 'Mapillary is specified as dataset in args but not config file'
        dataset_params = register_mapillary(
            use_color='no_color' not in args.dataset,
            ignore_other=args.data_no_other)
    if args.attack_type != 'none':
        main_attack(cfg, args, dataset_params)
    elif args.single_image:
        main_single(cfg, dataset_params)
    else:
        main(cfg, args)
