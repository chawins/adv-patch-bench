from .detectron.mapillary import get_mapillary_dict, register_mapillary
from .detectron.mapper import BenignMapper
from .detectron.mtsd import get_mtsd_dict, register_mtsd
from .image_only_dataset import ImageOnlyFolder
from .mtsd import MTSD

DATASET_DICT = {
    'mtsd': MTSD,
}


def load_dataset(args):
    loader = DATASET_DICT[args.dataset]['loader']
    return loader(args)
