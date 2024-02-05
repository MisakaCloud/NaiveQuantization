import logging
import math
import operator
import time

import torch

from util import AverageMeter, ProgressMeter

__all__ = ['train', 'validate', 'PerformanceScoreboard']

logger = logging.getLogger()


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def train(train_loader, model, quantizers, criterion, optimizer, quan_optimizer,
          scheduler, quan_scheduler, epoch, args):
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    data_time = AverageMeter('Time', ':6.3f')
    batch_time = AverageMeter('Data', ':6.3f')
    progress = ProgressMeter(
        len(train_loader),
        [losses, top1, top5, data_time, batch_time],
        prefix=f'Epoch: [{epoch}]')

    total_sample = len(train_loader.sampler)
    batch_size = train_loader.batch_size
    # steps_per_epoch = math.ceil(total_sample / batch_size)
    logger.info('Training: %d samples (%d per mini-batch)', total_sample, batch_size)

    model.train()
    end_time = time.time()
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end_time)

        # move data to the same device as model
        inputs = inputs.to(args.device.type, non_blocking=True)
        targets = targets.to(args.device.type, non_blocking=True)

        # compute output
        outputs = model(inputs)

        # compute loss
        loss = criterion(outputs, targets)
        # additive kd loss
        kd_loss = sum([quantizer.kd_loss for quantizer in quantizers])
        loss += kd_loss / len(quantizers) * 0.1

        # measure accuracy and record loss
        acc1, acc5 = accuracy(outputs.data, targets.data, topk=(1, 5))
        losses.update(loss.item(), inputs.size(0))
        top1.update(acc1.item(), inputs.size(0))
        top5.update(acc5.item(), inputs.size(0))

        scheduler.step(epoch=epoch, batch=batch_idx)
        quan_scheduler.step(epoch=epoch, batch=batch_idx)

        # compute gradient and update weights
        optimizer.zero_grad()
        quan_optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        quan_optimizer.step()

        batch_time.update(time.time() - end_time)
        end_time = time.time()

        if (batch_idx+1) % args.log.print_freq == 0:
            progress.display(batch_idx+1)

    if args.multiprocessing_distributed:
        top1.all_reduce()
        top5.all_reduce()
        losses.all_reduce()

    return top1.avg, top5.avg, losses.avg


def validate(val_loader, model, criterion, epoch, args):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(val_loader) + (args.distributed and (len(val_loader.sampler) * \
        args.world_size < len(val_loader.dataset))),
        [losses, top1, top5, batch_time],
        prefix='Test: ')

    def _run_validate(loader, base_progress=0):
        with torch.no_grad():
            end_time = time.time()
            for batch_idx, (inputs, targets) in enumerate(loader):
                batch_idx = base_progress + batch_idx
                inputs = inputs.to(args.device.type, non_blocking=True)
                targets = targets.to(args.device.type, non_blocking=True)

                outputs = model(inputs)
                loss = criterion(outputs, targets)

                acc1, acc5 = accuracy(outputs.data, targets.data, topk=(1, 5))
                losses.update(loss.item(), inputs.size(0))
                top1.update(acc1.item(), inputs.size(0))
                top5.update(acc5.item(), inputs.size(0))
                batch_time.update(time.time() - end_time)
                end_time = time.time()

                if (batch_idx+1) % args.log.print_freq == 0:
                    progress.display(batch_idx+1)

    total_sample = len(val_loader.sampler)
    batch_size = val_loader.batch_size
    # steps_per_epoch = math.ceil(total_sample / batch_size)

    logger.info('Validation: %d samples (%d per mini-batch)', total_sample, batch_size)

    model.eval()
    _run_validate(val_loader)

    if args.multiprocessing_distributed:
        top1.all_reduce()
        top5.all_reduce()
        losses.all_reduce()

    if args.multiprocessing_distributed and \
       (len(val_loader.sampler) * args.world_size < len(val_loader.dataset)):
        aux_dataset = Subset(val_loader.dataset,
                             range(len(val_loader.sampler) * args.world_size,
                                   len(val_loader.dataset)))
        aux_val_loader = torch.utils.data.DataLoader(
            aux_dataset, batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True)
        _run_validate(aux_val_loader, len(val_loader))

    # logger.info('==> Top1: %.3f    Top5: %.3f    Loss: %.3f\n',
    #             top1.avg, top5.avg, losses.avg)
    progress.display_summary()

    return top1.avg, top5.avg, losses.avg


class PerformanceScoreboard:
    def __init__(self, num_best_scores):
        self.board = list()
        self.num_best_scores = num_best_scores

    def update(self, top1, top5, epoch):
        """ Update the list of top training scores achieved so far, and log the best scores so far"""
        self.board.append({'top1': top1, 'top5': top5, 'epoch': epoch})

        # Keep scoreboard sorted from best to worst, and sort by top1, top5 and epoch
        curr_len = min(self.num_best_scores, len(self.board))
        self.board = sorted(self.board,
                            key=operator.itemgetter('top1', 'top5', 'epoch'),
                            reverse=True)[0:curr_len]
        for idx in range(curr_len):
            score = self.board[idx]
            logger.info('Scoreboard best %d ==> Epoch [%d][Top1: %.3f   Top5: %.3f]',
                        idx + 1, score['epoch'], score['top1'], score['top5'])

    def is_best(self, epoch):
        return self.board[0]['epoch'] == epoch

