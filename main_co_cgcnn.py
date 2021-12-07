import argparse
import os
import shutil
import sys
import time
import warnings
from random import sample

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn import metrics
from torch.autograd import Variable
from torch.optim.lr_scheduler import MultiStepLR

from co_cgcnn.data import CIFData
from co_cgcnn.data import collate_pool, get_train_val_loader
from co_cgcnn.model import Co_CrystalGraphConvNet

parser = argparse.ArgumentParser(description='Collaborative Crystal Graph Convolutional Neural Networks')

parser.add_argument('data_options', metavar='OPTIONS', nargs='+',
                    help='dataset options, started with the path to root dir, '
                         'then other options')
parser.add_argument('--disable-cuda', action='store_true',
                    help='Disable CUDA')
parser.add_argument('-j', '--workers', default=0, type=int, metavar='N',
                    help='number of data loading workers (default: 0)')
parser.add_argument('--epochs', default=200, type=int, metavar='N',
                    help='number of total epochs to run (default: 30)')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=512, type=int,
                    metavar='N', help='mini-batch size (default: 512)')
parser.add_argument('--lr', '--learning-rate', default=1e-2, type=float,
                    metavar='LR', help='initial learning rate (default: '
                                       '0.01)')
parser.add_argument('--lr-milestones', default=[50], nargs='+', type=int,
                    metavar='N', help='milestones for scheduler (default: '
                                      '[50])')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=0, type=float,
                    metavar='W', help='weight decay (default: 0)')
parser.add_argument('--print-freq', '-p', default=200, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
train_group = parser.add_mutually_exclusive_group()
train_group.add_argument('--train-ratio', default=0.8, type=float, metavar='N',
                    help='number of training data to be loaded (default none)')
valid_group = parser.add_mutually_exclusive_group()
valid_group.add_argument('--val-ratio', default=0.2, type=float, metavar='N',
                    help='percentage of validation data to be loaded (default '
                         '0.2)')

parser.add_argument('--optim', default='Adam', type=str, metavar='Adam',
                    help='choose an optimizer, SGD or Adam, (default: Adam)')
parser.add_argument('--atom-fea-len', default=64, type=int, metavar='N',
                    help='number of hidden atom features in conv layers')
parser.add_argument('--h-fea-len', default=64, type=int, metavar='N',
                    help='number of hidden features after pooling')
parser.add_argument('--p-fea-len', default=38, type=int, metavar='N',
                    help='number of hidden features after property embedding')
parser.add_argument('--n-prop', default=19, type=int, metavar='N',
                    help='number of properties')
parser.add_argument('--n-conv', default=4, type=int, metavar='N',
                    help='number of conv layers')
parser.add_argument('--n-h', default=5, type=int, metavar='N',
                    help='number of hidden layers after pooling')

args = parser.parse_args(sys.argv[1:])

print (args)

args.cuda = not args.disable_cuda and torch.cuda.is_available()

best_r2 = -1e10

def main():
    global args, best_r2

    # load data
    dataset = CIFData(*args.data_options)
    collate_fn = collate_pool
    train_loader, val_loader = get_train_val_loader(
        dataset=dataset,
        collate_fn=collate_fn,
        batch_size=args.batch_size,
        train_ratio=args.train_ratio,
        num_workers=args.workers,
        val_ratio=args.val_ratio,
        pin_memory=args.cuda)

#    print (dataset[0])

    structures, _, _ = dataset[0]
    orig_atom_fea_len = structures[0].shape[-1]
    nbr_fea_len = structures[1].shape[-1]
    model = Co_CrystalGraphConvNet(orig_atom_fea_len, nbr_fea_len,args.n_prop,
                                  atom_fea_len=args.atom_fea_len,
                                  n_conv=args.n_conv,
                                  h_fea_len=args.h_fea_len,
                                  n_h=args.n_h,
                                  p_fea_len = args.p_fea_len)
    if args.cuda:
        model.cuda()

    criterion = nn.MSELoss()

    if args.optim == 'SGD':
        optimizer = optim.SGD(model.parameters(), args.lr,
                              momentum=args.momentum,
                              weight_decay=args.weight_decay)
    elif args.optim == 'Adam':
        optimizer = optim.AdamW(model.parameters(), args.lr,
                               weight_decay=args.weight_decay)

    scheduler = MultiStepLR(optimizer, milestones=args.lr_milestones,
                            gamma=0.1)

    for epoch in range(args.start_epoch, args.epochs):
        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch)

        # evaluate on validation set
        val_r2 = validate(val_loader, model, criterion)

        if val_r2 != val_r2:
            print('Exit due to NaN')
            sys.exit(1)

        scheduler.step()

        # remember the best mae_eror and save checkpoint
        
        is_best = val_r2 > best_r2
        best_r2 = max(val_r2, best_r2)

        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_R2': best_r2,
            'optimizer': optimizer.state_dict(),
            'args': vars(args)
        }, is_best)

    # test best model
#    print('---------Evaluate Model on Test Set---------------')
#    best_checkpoint = torch.load('model_best.pth.tar')
#    model.load_state_dict(best_checkpoint['state_dict'])
#    validate(test_loader, model, criterion, normalizer, test=True)


def train(train_loader, model, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    naive_losses = AverageMeter()
    
    test_targets = []
    test_preds = []

    model.train()

    end = time.time()
    for i, (input, target, _) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if args.cuda:
            input_var = (Variable(input[0].cuda(non_blocking=True)),
                         Variable(input[1].cuda(non_blocking=True)),
                         Variable(input[2].cuda(non_blocking=True)),
                         input[3].cuda(non_blocking=True),
                         [crys_idx.cuda(non_blocking=True) for crys_idx in input[4]])
        else:
            input_var = (Variable(input[0]),
                         Variable(input[1]),
                         Variable(input[2]),
                         input[3],
                         input[4])

        target_var = Variable(target.cuda(non_blocking=True))

        output = model(*input_var)
        loss = criterion(output.view(-1), target_var.view(-1))
        naive_loss = criterion(torch.ones(target.size(0))*torch.mean(target).item(), target)

        losses.update(loss.data.cpu(), target.size(0))
        naive_losses.update(naive_loss, target.size(0))

        test_pred = output.data.cpu()
        test_target = target
        test_preds += test_pred.view(-1).tolist()
        test_targets += test_target.view(-1).tolist()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Naive_Loss {naive_loss.val:.4f} ({naive_loss.avg:.4f})\t'.format(
                    epoch, i, len(train_loader), batch_time=batch_time,
                    data_time=data_time, loss=losses, naive_loss=naive_losses)
                )
            print ('Cumulative R2 score: %f'%(metrics.r2_score(test_targets, test_preds)))

def validate(val_loader, model, criterion):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    naive_losses = AverageMeter()

    test_targets = []
    test_preds = []

    model.eval()

    end = time.time()
    for i, (input, target, batch_cif_ids) in enumerate(val_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if args.cuda:
            with torch.no_grad():
                input_var = (Variable(input[0].cuda(non_blocking=True)),
                             Variable(input[1].cuda(non_blocking=True)),
                             Variable(input[2].cuda(non_blocking=True)),
                             input[3].cuda(non_blocking=True),
                             [crys_idx.cuda(non_blocking=True) for crys_idx in input[4]])
        else:
            with torch.no_grad():
                input_var = (Variable(input[0]),
                             Variable(input[1]),
                             Variable(input[2]),
                             input[3],
                             input[4])
        target_var = Variable(target.cuda(non_blocking=True))

        output = model(*input_var)
        loss = criterion(output.view(-1), target_var.view(-1))
        naive_loss = criterion(torch.ones(target.size(0))*torch.mean(target).item(), target)
        
        losses.update(loss.data.cpu(), target.size(0))
        naive_losses.update(naive_loss, target.size(0))


        test_pred = output.data.cpu()
        test_target = target
        test_preds += test_pred.view(-1).tolist()
        test_targets += test_target.view(-1).tolist()

        batch_time.update(time.time() - end)
        end = time.time()
        
        if i % args.print_freq == 0:
            print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Naive_Loss {naive_loss.val:.4f} ({naive_loss.avg:.4f})\t'.format(
                    i, len(val_loader), batch_time=batch_time,
                    data_time=data_time, loss=losses, naive_loss=naive_losses)
                )

    print ('Cumulative R2 score: %f'%(metrics.r2_score(test_targets, test_preds)))

    return metrics.r2_score(test_targets, test_preds)

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

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


def adjust_learning_rate(optimizer, epoch, k):
    """Sets the learning rate to the initial LR decayed by 10 every k epochs"""
    assert type(k) is int
    lr = args.lr * (0.1 ** (epoch // k))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def r2_score(output, target):
    target_mean = torch.mean(target)
    ss_tot = torch.sum((target - target_mean) ** 2)
    ss_res = torch.sum((target - output) ** 2)
    r2 = 1 - ss_res / ss_tot
    return r2

if __name__ == '__main__':
    main()
