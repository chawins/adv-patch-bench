from .image_only_dataset import ImageOnlyFolder
from .mtsd import MTSD

DATASET_DICT = {
    'mtsd': MTSD,
}


def load_dataset(args):
    loader = DATASET_DICT[args.dataset]['loader']
    return loader(args)
