import cv2
from detectron2.config import get_cfg
from detectron2.data import (MetadataCatalog, build_detection_test_loader,
                             build_detection_train_loader)
from detectron2.engine import (DefaultPredictor, default_argument_parser,
                               default_setup)
from detectron2.evaluation import inference_on_dataset, verify_results
from detectron2.utils.visualizer import Visualizer

import adv_patch_bench.utils.detectron.custom_coco_evaluator as cocoeval
from adv_patch_bench.dataloaders.mtsd_detectron import (get_mtsd_dict,
                                                        register_mtsd)
from hparams import DATASETS, NUM_CLASSES, OTHER_SIGN_CLASS


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


def main_single(cfg, args, dataset_params):
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
        print(inpt['file_name'])
        import pdb
        pdb.set_trace()
        # img = inpt[0]['image'].permute(1, 2, 0).numpy()
        # prediction = model(img)
        # visualizer = Visualizer(img[:, :, ::-1], metadata=metadata, scale=0.5)
        # out_gt = visualizer.draw_dataset_dict(inpt[0])
        img = cv2.imread(inpt['file_name'])
        prediction = model(img)
        visualizer = Visualizer(img[:, :, ::-1], metadata=metadata, scale=0.5)
        out_gt = visualizer.draw_dataset_dict(inpt)
        out_gt.save('gt.png')
        visualizer = Visualizer(img[:, :, ::-1], metadata=metadata, scale=0.5)
        out_pred = visualizer.draw_instance_predictions(prediction['instances'].to('cpu'))
        out_pred.save('pred.png')


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)

    # Additional custom setup
    no_other = 'orig' not in args.dataset and args.data_no_other
    num_classes = NUM_CLASSES[args.dataset] - no_other
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_classes
    print(f'=> Using {args.dataset} with {num_classes} thing classes.')

    cfg.freeze()
    default_setup(cfg, args)
    return cfg


if __name__ == "__main__":
    parser = default_argument_parser()
    parser.add_argument('--single-image', action='store_true')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--data-no-other', action='store_true',
                        help='If True, do not load "other" or "background" class to the dataset.')
    parser.add_argument('--eval-mode', type=str, default='default')
    args = parser.parse_args()
    print('Command Line Args:', args)

    # Verify some args
    assert args.dataset in DATASETS
    cfg = setup(args)

    # Register dataset
    dataset_params = register_mtsd(
        use_mtsd_original_labels='orig' in args.dataset,
        use_color='no_color' not in args.dataset,
        ignore_other=args.data_no_other)
    if args.single_image:
        main_single(cfg, args, dataset_params)
    else:
        main(cfg, args)
