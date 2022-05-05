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
from hparams import DATASETS, OTHER_SIGN_CLASS, LABEL_LIST


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
    parser = default_argument_parser()
    parser.add_argument('--single-image', action='store_true')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--data-no-other', action='store_true',
                        help='If True, do not load "other" or "background" class to the dataset.')
    parser.add_argument('--eval-mode', type=str, default='default')

    parser.add_argument('--interp', type=str, default='bilinear',
                        help='interpolation method (nearest, bilinear, bicubic)')
    parser.add_argument('--synthetic-eval', action='store_true',
                        help='evaluate with pasted synthetic signs')
    parser.add_argument('--name', default='exp', help='save to project/name')
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--seed', type=int, default=0, help='set random seed')
    parser.add_argument('--padded-imgsz', type=str, default='3000,4000',
                        help='final image size including padding (height,width). Default: 3000,4000')

    # =========================== Attack arguments ========================== #
    parser.add_argument('--attack-type', type=str, required=True,
                        help='which attack evaluation to run (none, load, per-sign, random, debug)')
    parser.add_argument('--adv-patch-path', type=str, default='',
                        help='path to adv patch and mask to load')
    parser.add_argument('--obj-class', type=int, default=0, help='class of object to attack')
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

    args = parser.parse_args()
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
