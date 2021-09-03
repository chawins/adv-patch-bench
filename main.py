#!/usr/bin/env python
# The template of this code is generously provided by Norman Mu (@normster)
import argparse
import json
import math
import os
import sys
import time

import numpy as np
import timm
import torch
import torch.backends.cudnn as cudnn
import torch.cuda.amp as amp
import torch.distributed as dist
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import wandb

import utils


def get_args_parser():
    parser = argparse.ArgumentParser(description='Simple ImageNet classification', add_help=False)
    parser.add_argument('--data', default='/home/owner/data/auto3d/041221', type=str)
    parser.add_argument('--arch', default='resnet50', type=str)
    parser.add_argument('--output-dir', default='./', type=str, help='output dir')
    parser.add_argument('-j', '--workers', default=10, type=int, metavar='N',
                        help='number of data loading workers per process')
    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--start-epoch', default=0, type=int)
    parser.add_argument('--batch-size', default=256, type=int,
                        help='mini-batch size per device.')
    parser.add_argument('--full-precision', action='store_true')
    parser.add_argument('--warmup-epochs', default=0, type=int)
    parser.add_argument('--lr', default=0.1, type=float)
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--wd', default=1e-4, type=float)
    parser.add_argument('--optim', default='sgd', type=str)
    parser.add_argument('--betas', default=(0.9, 0.999), nargs=2, type=float)
    parser.add_argument('--eps', default=1e-8, type=float)
    parser.add_argument('--print-freq', default=10, type=int, help='print frequency')
    parser.add_argument('--resume', default='', type=str, help='path to latest checkpoint')
    parser.add_argument('--evaluate', action='store_true', help='Evaluate only')
    parser.add_argument('--world-size', default=1, type=int,
                        help='number of nodes for distributed training')
    parser.add_argument('--rank', default=0, type=int,
                        help='node rank for distributed training')
    parser.add_argument('--dist-url', default='tcp://localhost:10001', type=str,
                        help='url used to set up distributed training')
    parser.add_argument('--dist-backend', default='nccl', type=str)
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--gpu', default=None, type=int, help='GPU id to use.')
    parser.add_argument('--wandb', action='store_true', help='Enable WandB')
    return parser


best_acc1 = 0


# dist eval introduces slight variance to results if
# num samples != 0 mod (batch size * num gpus)
def get_loader_sampler(root, transform, args):
    dataset = datasets.ImageFolder(root, transform=transform)
    if args.distributed:
        sampler = torch.utils.data.distributed.DistributedSampler(dataset)
    else:
        sampler = None
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=(sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=sampler, drop_last=True)

    return loader, sampler


def main(args):
    utils.init_distributed_mode(args)

    global best_acc1

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    # create model
    print("=> creating model")
    model = timm.create_model(args.arch, num_classes=24)
    model.cuda(args.gpu)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)

    p_wd, p_non_wd = [], []
    for n, p in model.named_parameters():
        if 'bias' in n or 'ln' in n or 'bn' in n:
            p_non_wd.append(p)
        else:
            p_wd.append(p)

    optim_params = [{"params": p_wd, "weight_decay": args.wd},
                    {"params": p_non_wd, "weight_decay": 0}]

    if args.optim == 'sgd':
        optimizer = torch.optim.SGD(optim_params, lr=args.lr,
                                    momentum=args.momentum, weight_decay=args.wd)
    else:
        optimizer = torch.optim.AdamW(optim_params, lr=args.lr, betas=args.betas,
                                      eps=args.eps, weight_decay=args.wd)

    scaler = amp.GradScaler(enabled=not args.full_precision)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading resume checkpoint '{}'".format(args.resume))
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
            print("=> loaded resume checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    # resume from latest checkpoint in output directory
    latest = os.path.join(args.output_dir, 'checkpoint.pt')
    if os.path.exists(latest):
        print("=> loading latest checkpoint '{}'".format(latest))
        if args.gpu is None:
            checkpoint = torch.load(latest)
        else:
            # Map model to be loaded to specified single gpu.
            loc = 'cuda:{}'.format(args.gpu)
            checkpoint = torch.load(latest, map_location=loc)
        args.start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scaler.load_state_dict(checkpoint['scaler'])
        best_acc1 = checkpoint['best_acc1']
        print("=> loaded latest checkpoint '{}' (epoch {})"
              .format(latest, checkpoint['epoch']))

    cudnn.benchmark = True

    # Data loading code
    print("=> creating dataset")
    normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                     std=[0.5, 0.5, 0.5])
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(128, scale=(0.5, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ])
    val_transform = transforms.Compose([
        transforms.Resize(int(128 * 256 / 224)),
        transforms.CenterCrop(128),
        transforms.ToTensor(),
        normalize
    ])

    train_loader, train_sampler = get_loader_sampler(os.path.join(args.data, 'train'),
                                                     train_transform, args)
    val_loader, _ = get_loader_sampler(os.path.join(args.data, 'val'),
                                       val_transform, args)

    if args.evaluate:
        validate(val_loader, model, criterion, args)
        return

    if utils.is_main_process() and args.wandb:
        wandb_id = os.path.split(args.output_dir)[-1]
        wandb.init(entity='auto3d', project='auto3d', id=wandb_id, config=args, resume='allow')
        print('wandb step:', wandb.run.step)

    print(args)

    print("=> beginning training")
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        lr = adjust_learning_rate(optimizer, epoch, args)
        print(f"=> lr @ epoch {epoch}: {lr:.2e}")

        # train for one epoch
        train_stats = train(train_loader, model, criterion, optimizer, scaler, epoch, args)
        val_stats = validate(val_loader, model, criterion, args)
        acc1 = val_stats['acc1']

        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)

        print("=> saving checkpoint")
        utils.save_on_master({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scaler': scaler.state_dict(),
            'best_acc1': best_acc1,
            'args': args,
        }, is_best, args.output_dir)

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     **{f'test_{k}': v for k, v in val_stats.items()},
                     'epoch': epoch}

        if utils.is_main_process():
            if args.wandb:
                wandb.log(log_stats)
            with open(os.path.join(args.output_dir, 'log.txt'), 'a') as f:
                f.write(json.dumps(log_stats) + '\n')


def train(train_loader, model, criterion, optimizer, scaler, epoch, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    mem = AverageMeter('Mem (GB)', ':6.1f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5, mem],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    for i, (images, targets) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        images = images.cuda(args.gpu, non_blocking=True)
        targets = targets.cuda(args.gpu, non_blocking=True)

        # compute output
        with amp.autocast(enabled=not args.full_precision):
            outputs = model(images)
            loss = criterion(outputs, targets)

        if not math.isfinite(loss.item()):
            print("Loss is {}, stopping training".format(loss.item()))
            sys.exit(1)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1.item(), images.size(0))
        top5.update(acc5.item(), images.size(0))

        # compute gradient and do SGD step
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        mem.update(torch.cuda.max_memory_allocated() // 1e9)

        if i % args.print_freq == 0:
            if utils.is_main_process() and args.wandb:
                wandb.log({'acc': acc1.item(), 'loss': loss.item(), 'scaler': scaler.get_scale()})
            progress.display(i)

    progress.synchronize()
    return {'acc1': top1.avg, 'acc5': top5.avg, 'loss': losses.avg,
            'lr': optimizer.param_groups[0]['lr']}


def validate(val_loader, model, criterion, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    mem = AverageMeter('Mem (GB)', ':6.1f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, data_time, losses, top1, top5, mem],
        prefix="Test: ")

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (images, targets) in enumerate(val_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        images = images.cuda(args.gpu, non_blocking=True)
        targets = targets.cuda(args.gpu, non_blocking=True)

        # compute output
        with torch.no_grad():
            outputs = model(images)
            loss = criterion(outputs, targets)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1.item(), images.size(0))
        top5.update(acc5.item(), images.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        mem.update(torch.cuda.max_memory_allocated() // 1e9)

        if i % args.print_freq == 0:
            progress.display(i)

    # TODO: this should also be done with the ProgressMeter
    print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
          .format(top1=top1, top5=top5))

    progress.synchronize()
    return {'acc1': top1.avg, 'acc5': top5.avg, 'loss': losses.avg}


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def synchronize(self):
        t = torch.tensor([self.sum, self.count], dtype=torch.float64, device='cuda')
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.sum = int(t[0])
        self.count = t[1]
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def synchronize(self):
        for meter in self.meters:
            meter.synchronize()

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def adjust_learning_rate(optimizer, epoch, args):
    """Decay the learning rate based on schedule"""
    lr = args.lr
    if epoch < args.warmup_epochs:
        lr *= (epoch + 1) / args.warmup_epochs
    lr *= 0.5 * (1. + math.cos(math.pi * epoch / args.epochs))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.reshape(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Simple ImageNet classification', parents=[get_args_parser()])
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    main(args)
