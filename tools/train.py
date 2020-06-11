from __future__ import print_function, division, absolute_import
import argparse
import os
import shutil
import time

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from collections import OrderedDict
from efficientnet_pytorch import EfficientNet

import sys
sys.path.append('.')
from utils.utils import SceneData, FocalLoss, mixup
from utils.model import create_model
from mmcv import Config
from utils.transform import image_transforms

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('config', help='train config file path')

args = parser.parse_args()
cfg = Config.fromfile(args.config)

def main():
    # global args, best_prec1, cfg 
    model = create_model(cfg.model.arch, cfg.model.num_classes, cfg.model.pretrained)

    best_prec1 = 0
    # optionally resume from a checkpoint
    if cfg.resume:
        if os.path.isfile(cfg.resume):
            print("=> loading checkpoint '{}'".format(cfg.resume))
            checkpoint = torch.load(cfg.resume)
            cfg.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            state_dict = checkpoint['state_dict']
            
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k[7:]
                new_state_dict[name] = v

            model.load_state_dict(new_state_dict)
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(cfg.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(cfg.resume))

    cudnn.benchmark = True

    # Data loading code
    print("Loading data...")
    traindir = os.path.join(cfg.data, 'train')
    valdir = os.path.join(cfg.data, 'val')

    train_loader = torch.utils.data.DataLoader(
        SceneData(txt_file=cfg.annotations.train, 
                    image_dir=cfg.data, 
                    mode='train', 
                    transform=image_transforms['train']),
        batch_size=cfg.batch_size, shuffle=True,
        num_workers=cfg.workers, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
            SceneData(txt_file=cfg.annotations.val,
                    image_dir=cfg.data, 
                    mode='train', 
                    transform=image_transforms['val']),
        batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.workers, pin_memory=True)

    # define loss function (criterion) and optimizer
    # criterion = nn.CrossEntropyLoss().cuda()
    criterion = FocalLoss(class_num=cfg.model.num_classes)

    optimizer = optim.SGD(model.parameters(), cfg.optimizer.learning_rate,
                                momentum=cfg.optimizer.momentum,
                                weight_decay=cfg.optimizer.weight_decay)


    model = torch.nn.DataParallel(model, device_ids=cfg.device_ids).cuda()
    # model = model.cuda()

    if cfg.evaluate:
        validate(val_loader, model, criterion)
        return

    for epoch in range(cfg.start_epoch, cfg.total_epochs):
        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch)

        # evaluate on validation set
        prec1 = validate(val_loader, model, criterion)

        # remember best prec@1 and save checkpoint
        is_best = prec1[0] > best_prec1
        best_prec1 = max(prec1[0], best_prec1)
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': cfg.model.arch,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
        }, epoch, is_best, cfg.work_dir)


def train(train_loader, model, criterion, optimizer, epoch):
    
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (inputs, targets) in enumerate(train_loader):
        adjust_learning_rate(optimizer, cfg.optimizer.learning_rate, epoch)
        # measure data loading time
        data_time.update(time.time() - end)

        # targets = targets.cuda()
        # inputs = inputs.cuda()
        # inputs_var = torch.autograd.Variable(inputs)
        # targets_var = torch.autograd.Variable(targets)

        # # compute output
        # outputs = model(inputs_var)
        # loss = criterion(outputs, targets_var)

        # # measure accuracy and record loss
        # prec1, prec5 = accuracy(outputs.data, targets, topk=(1, 5))

        # mixup
        mixed_images, labels_a, labels_b, seed = mixup(inputs.numpy(), targets.numpy())
        inputs = torch.from_numpy(mixed_images).cuda()
        labels_a, labels_b = torch.from_numpy(labels_a).cuda(), torch.from_numpy(labels_b).cuda()

        outpus = model(inputs)
        loss = seed * criterion(outpus, labels_a) + (1-seed) * criterion(outpus, labels_b)
        
        prec1_1, prec1_5 = accuracy(outpus.data, labels_a, topk=(1, 5))
        prec2_1, prec2_5 = accuracy(outpus.data, labels_b, topk=(1, 5))
        prec1 = seed * prec1_1 + (1 - seed) * prec2_1
        prec5 = seed * prec1_5 + (1 - seed) * prec2_5

        losses.update(loss.data.item(), inputs.size(0))
        top1.update(prec1.item(), inputs.size(0))
        top5.update(prec5.item(), inputs.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % cfg.log_config.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Learning Rate {learning_rate:.4f}\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                epoch, i, len(train_loader), batch_time=batch_time,
                learning_rate=optimizer.param_groups[0]['learning_rate'], loss=losses, top1=top1, top5=top5))


def validate(val_loader, model, criterion):
    with torch.no_grad():
        batch_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()

        # switch to evaluate mode
        model.eval()

        end = time.time()
        for i, (inputs, targets) in enumerate(val_loader):
            targets = targets.cuda()
            inputs = inputs.cuda()

            # compute output
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))
            losses.update(loss.data.item(), inputs.size(0))
            top1.update(prec1.item(), inputs.size(0))
            top5.update(prec5.item(), inputs.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % cfg.log_config.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                       i, len(val_loader), batch_time=batch_time, loss=losses,
                       top1=top1, top5=top5))

        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))

        return top1.avg, top5.avg


def save_checkpoint(state, epoch, is_best, config, filename='checkpoint.pth.tar'):
    try:
        os.makedirs(config)
    except:
        pass
    torch.save(state, os.path.join(config, 'epoch_{:03d}.pth.tar'.format(epoch)))
    torch.save(state, os.path.join(config, filename))
    if is_best:
        shutil.copyfile(os.path.join(config, filename), os.path.join(config, 'model_best.pth.tar'))

def adjust_learning_rate(optimizer, learning_rate, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    learning_rate = learning_rate * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['learning_rate'] = learning_rate

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
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


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


if __name__ == '__main__':
    main()

