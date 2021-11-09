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
from torch.utils.data import DataLoader

from co_cgcnn.data import CIFData
from co_cgcnn.data import collate_pool
from co_cgcnn.model import Co_CrystalGraphConvNet

parser = argparse.ArgumentParser(description='Collaborative Crystal Graph Convolutional Neural Networks')

parser.add_argument('modelpath', help='path to the trained model.')
parser.add_argument('cifpath', help='path to the directory of CIF files.')
parser.add_argument('-b', '--batch-size', default=512, type=int,
                    metavar='N', help='mini-batch size (default: 512)')
parser.add_argument('-j', '--workers', default=0, type=int, metavar='N',
                    help='number of data loading workers (default: 0)')
parser.add_argument('--disable-cuda', action='store_true',
                    help='Disable CUDA')
parser.add_argument('--print-freq', '-p', default=100, type=int,
                    metavar='N', help='print frequency (default: 100)')

args = parser.parse_args(sys.argv[1:])

print (args)

if os.path.isfile(args.modelpath):
    print("=> loading model params '{}'".format(args.modelpath))
    model_checkpoint = torch.load(args.modelpath,
                                  map_location=lambda storage, loc: storage)
    model_args = argparse.Namespace(**model_checkpoint['args'])
    print("=> loaded model params '{}'".format(args.modelpath))
else:
    print("=> no model params found at '{}'".format(args.modelpath))


args.cuda = not args.disable_cuda and torch.cuda.is_available()

best_loss = 1e10

def main():
    global args, model_args, best_loss

    # load data
    dataset = CIFData(args.cifpath)
    collate_fn = collate_pool
    test_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                             num_workers=args.workers, collate_fn=collate_fn,
                             pin_memory=args.cuda)

#    print (dataset[0])

    structures, _, _ = dataset[0]
    orig_atom_fea_len = structures[0].shape[-1]
    nbr_fea_len = structures[1].shape[-1]
    model = Co_CrystalGraphConvNet(orig_atom_fea_len, nbr_fea_len, model_args.n_prop,
                                  atom_fea_len=model_args.atom_fea_len,
                                  n_conv=model_args.n_conv,
                                  h_fea_len=model_args.h_fea_len,
                                  n_h=model_args.n_h,
                                  p_fea_len = model_args.p_fea_len)
    if args.cuda:
        model.cuda()

    criterion = nn.MSELoss()

    print("=> loading model '{}'".format(args.modelpath))
    checkpoint = torch.load(args.modelpath,
                                map_location=lambda storage, loc: storage)
    model.load_state_dict(checkpoint['state_dict'])
    print("=> loaded model '{}' (epoch {}, validation {})"
              .format(args.modelpath, checkpoint['epoch'],
                      checkpoint['best_R2']))

    test(test_loader, model, criterion)


def test(val_loader, model, criterion):
    batch_time = AverageMeter()
    losses = AverageMeter()
    data_time = AverageMeter()
    naive_losses = AverageMeter()

    test_targets = []
    test_preds = []
    test_cif_ids = []
    test_props = []
    
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
        loss = criterion(output, target_var)
        naive_loss = criterion(torch.ones(target.size(0))*torch.mean(target).item(), target)
        
        losses.update(loss.data.cpu(), target.size(0))
        naive_losses.update(naive_loss, target.size(0))

        prop = input[2].data.cpu().numpy()
        for p in prop:
            test_props.append(np.where(p==1)[0][0])


        test_pred = output.data.cpu()
        test_target = target
        test_preds += test_pred.view(-1).tolist()
        test_targets += test_target.view(-1).tolist()
        test_cif_ids += batch_cif_ids

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
    
    star_label = '**'
    import csv
    with open('test_results.csv', 'w') as f:
        writer = csv.writer(f)
        for cif_id, prop, target, pred in zip(test_cif_ids, test_props, test_targets,
                                            test_preds):
            writer.writerow((cif_id, prop, target, pred))

    print(' {star} Loss {losses.avg:.3f}'.format(star=star_label,
                                                        losses=losses))
    print (float(losses.avg/naive_losses.avg))
    return float(losses.avg/naive_losses.avg)

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
