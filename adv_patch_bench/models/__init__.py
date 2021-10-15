import os

import timm
import torch
import torch.cuda.amp as amp
import torch.nn as nn

from adv_patch_bench.dataloaders import DATASET_DICT
from .common import Normalize


def build_classifier(args):

    assert args.dataset in DATASET_DICT

    normalize = DATASET_DICT[args.dataset]['normalize']
    model = timm.create_model(args.arch, pretrained=args.pretrained, num_classes=0)
    with torch.no_grad():
        dummy_input = torch.zeros((2, ) + DATASET_DICT[args.dataset]['input_dim'])
        rep_dim = model(dummy_input).size(-1)

    model.fc = nn.Linear(rep_dim, args.num_classes)
    model = nn.Sequential(Normalize(**normalize), model)

    n_model = sum(p.numel() for p in model.parameters()) / 1e6
    print(f'=> total params: {n_model:.2f}M')

    model.cuda(args.gpu)

    if args.distributed:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])

    p_wd, p_non_wd = [], []
    for n, p in model.named_parameters():
        if p.requires_grad:
            if 'bias' in n or 'ln' in n or 'bn' in n:
                p_non_wd.append(p)
            else:
                p_wd.append(p)

    optim_params = [{'params': p_wd, 'weight_decay': args.wd},
                    {'params': p_non_wd, 'weight_decay': 0}]

    if args.optim == 'sgd':
        optimizer = torch.optim.SGD(optim_params, lr=args.lr,
                                    momentum=args.momentum, weight_decay=args.wd)
    else:
        optimizer = torch.optim.AdamW(optim_params, lr=args.lr, betas=args.betas,
                                      eps=args.eps, weight_decay=args.wd)

    scaler = amp.GradScaler(enabled=not args.full_precision)

    # Optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print(f'=> loading resume checkpoint {args.resume}')
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            scaler.load_state_dict(checkpoint['scaler'])
            best_acc1 = checkpoint['best_acc1']
            print(f'=> loaded resume checkpoint (epoch {checkpoint["epoch"]})')
        else:
            print(f'=> no checkpoint found at {args.resume}')

    return model, optimizer, scaler
