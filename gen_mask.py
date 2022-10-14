import os

import matplotlib.pyplot as plt
import numpy as np
import torchvision
from PIL import Image

from adv_patch_bench.attacks.attack_util import get_object_and_mask_from_numpy
from hparams import LABEL_LIST

from adv_patch_bench import utils




if __name__ == "__main__":
    # parser = argparse.ArgumentParser(
    #     description="Mask Generation", add_help=False
    # )
    # parser.add_argument(
    #     "--syn-obj-path",
    #     type=str,
    #     default="",
    #     help="path to synthetic image of the object",
    # )
    # parser.add_argument("--obj-size", type=int, required=True)
    # # parser.add_argument('--patch-size', type=int, required=True)
    # parser.add_argument("--patch-name", type=str, default=None)
    # parser.add_argument("--dataset", type=str, required=True)
    # parser.add_argument(
    #     "--obj-class",
    #     type=int,
    #     default=-1,
    #     help="class of object to attack (-1: all classes)",
    # )
    # parser.add_argument("--save-mask", action="store_true")
    # args = parser.parse_args()
    # parse_dataset_name(args)
    config = utils.argparse.eval_args_parser(True)
    cfg = utils.argparse.setup_detectron_test_args(config)

    # Get obj size in inch based on given obj class
    class_names = LABEL_LIST[args.dataset]
    obj_width_inch = get_obj_width(args.obj_class, class_names)
    obj_numpy = np.array(Image.open(args.syn_obj_path).convert("RGBA")) / 255
    patch_mask = gen_patch_mask(obj_numpy, args.obj_size, obj_width_inch)

    if args.save_mask and args.patch_name is not None:
        save_dir = "./masks/"
        os.makedirs(save_dir, exist_ok=True)
        mask_save_path = os.path.join(save_dir, f"{args.patch_name}.png")
        print(f"=> Saving patch mask to {mask_save_path}")
        torchvision.utils.save_image(patch_mask[0], mask_save_path)

        # plt.imshow(patch_mask[0], cmap='gray')
        # plt.savefig(mask_save_path, bbox_inches='tight')
        # plt.close()

        obj, _ = get_object_and_mask_from_numpy(obj_numpy, patch_mask.shape[1:])
        plot_image = obj * (1 - patch_mask)
        plt.imshow(plot_image.permute(1, 2, 0).clamp(0, 1))
        mask_save_path = os.path.join(
            save_dir, f"{args.patch_name}_on_sign.jpg"
        )
        plt.savefig(mask_save_path, bbox_inches="tight")
        plt.close()
