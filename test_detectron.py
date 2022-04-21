import cv2
from detectron2.config import get_cfg
from detectron2.data import (MetadataCatalog, build_detection_test_loader,
                             build_detection_train_loader)
from detectron2.engine import (DefaultPredictor, default_argument_parser,
                               default_setup)
from detectron2.evaluation import inference_on_dataset, verify_results
from detectron2.utils.visualizer import Visualizer

# Import this file to register MTSD for detectron
from adv_patch_bench.dataloaders.mtsd_detectron import get_mtsd_dict
from adv_patch_bench.utils.custom_coco_evaluator import CustomCOCOEvaluator


def main(cfg, args):
    # distributed is set to False
    evaluator = CustomCOCOEvaluator('mtsd_val', cfg, False,
                                    output_dir=cfg.OUTPUT_DIR,
                                    use_fast_impl=True)
    if args.debug:
        sampler = list(range(100))
    else:
        sampler = None
    val_loader = build_detection_test_loader(cfg, 'mtsd_val',
                                             # batch_size=cfg.SOLVER.IMS_PER_BATCH,
                                             batch_size=1,
                                             num_workers=cfg.DATALOADER.NUM_WORKERS,
                                             sampler=sampler)
    predictor = DefaultPredictor(cfg)
    print(inference_on_dataset(predictor.model, val_loader, evaluator))


def main_single(cfg, args):
    # Build model
    model = DefaultPredictor(cfg)
    # Build dataloader
    metadata = MetadataCatalog.get(cfg.DATASETS.TEST[0])
    # val_loader = build_detection_test_loader(cfg, cfg.DATASETS.TEST[0],
    #                                          batch_size=1,
    #                                          num_workers=cfg.DATALOADER.NUM_WORKERS)
    val_loader = build_detection_train_loader(cfg)
    # val_loader = get_mtsd_dict('val')
    for i, inpt in enumerate(val_loader):
        import pdb
        pdb.set_trace()
        # img = inpt[0]['image'].permute(1, 2, 0).numpy()
        # prediction = model(img)
        # visualizer = Visualizer(img[:, :, ::-1], metadata=metadata, scale=0.5)
        # out_gt = visualizer.draw_dataset_dict(inpt[0])
        print(inpt['file_name'])
        img = cv2.imread(inpt['file_name'])
        prediction = model(img)
        visualizer = Visualizer(img[:, :, ::-1], metadata=metadata, scale=0.5)
        out_gt = visualizer.draw_dataset_dict(inpt)
        out_gt.save('gt.png')
        out_pred = visualizer.draw_instance_predictions(prediction['instances'].to('cpu'))
        out_pred.save('pred.png')
        import pdb
        pdb.set_trace()
        # HVPLEUNAPVajL8gJaM-sSw.txt


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
    parser = default_argument_parser()
    parser.add_argument('--single-image', action='store_true')
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()
    print("Command Line Args:", args)
    cfg = setup(args)
    if args.single_image:
        main_single(cfg, args)
    else:
        main(cfg, args)
