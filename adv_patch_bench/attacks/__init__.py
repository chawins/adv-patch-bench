import torch.nn as nn
from adv_patch_bench.utils import KLDLoss, PixelwiseCELoss

from .auto import AutoAttackModule
from .none import NoAttackModule
from .pgd import PGDAttackModule
from .trades import TRADESAttackModule


def get_ce_loss(args):
    if args.experiment == 'seg-only':
        loss = PixelwiseCELoss()
    else:
        loss = nn.CrossEntropyLoss(reduction='none')
    return loss.cuda(args.gpu)


def setup_eval_attacker(args, model, num_classes=None):

    if num_classes is None:
        num_classes = args.num_classes
    eps = float(args.epsilon)
    norm = args.atk_norm
    attack_config = {
        'pgd_steps': 10,
        'pgd_step_size': eps / 4,
        'num_restarts': 1,
    }

    loss = get_ce_loss(args)
    pgd_attack = PGDAttackModule(attack_config, model, loss, norm, eps)
    no_attack = NoAttackModule(None, None, None, norm, eps)
    auto_attack = AutoAttackModule(None, model, None, norm, eps,
                                   verbose=True, num_classes=num_classes)

    return no_attack, pgd_attack, auto_attack


def setup_train_attacker(args, model):

    eps = float(args.epsilon)
    norm = args.atk_norm
    attack_config = {
        'pgd_steps': 10,
        'pgd_step_size': eps / 4,
        'num_restarts': 1,
    }

    if args.adv_train == 'none':
        attack = NoAttackModule(None, None, None, norm, eps)
    elif args.adv_train == 'pgd':
        loss = get_ce_loss(args)
        attack = PGDAttackModule(attack_config, model, loss, norm, eps)
    elif args.adv_train == 'trades':
        loss = KLDLoss(reduction='sum-non-batch').cuda(args.gpu)
        attack = TRADESAttackModule(attack_config, model, loss, norm, eps)
    else:
        raise NotImplementedError('adv-train is not recognized.')

    return attack
