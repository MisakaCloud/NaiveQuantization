import torch
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter

import logging
from enum import Enum

__all__ = ['ProgressMeter', 'ProgressMonitor', 'TensorBoardMonitor', 'AverageMeter']

logger = logging.getLogger()

class Summary(Enum):
    NONE = 0
    AVERAGE = 1
    SUM = 2
    COUNT = 3


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt='%.6f', summary_type=Summary.AVERAGE):
        self.name = name
        self.fmt = fmt
        self.summary_type = summary_type
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

    def all_reduce(self):
        if torch.cuda.is_available():
            device = torch.device('cuda')
        elif torch.backends.mps.is_available():
            device = torch.device('mps')
        else:
            device = torch.device('cpu')
        total = torch.tensor([self.sum, self.count], dtype=torch.float32, device=device)
        dist.all_reduce(total, dist.ReduceOp.SUM, async_op=False)
        self.sum, self.count = total.tolist()
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'

        return fmtstr.format(**self.__dict__)

    def summary(self):
        fmtstr = ''
        if self.summary_type is Summary.NONE:
            fmtstr = ''
        elif self.summary_type is Summary.AVERAGE:
            fmtstr = '{name} {avg:.3f}'
        elif self.summary_type is Summary.SUM:
            fmtstr = '{name} {sum:.3f}'
        elif self.summary_type is Summary.COUNT:
            fmtstr = '{name} {count:.3f}'
        else:
            raise ValueError('invalid summary type %r' % self.summary_type)

        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        logger.info('\t'.join(entries))

    def display_summary(self):
        entries = [" *"]
        entries += [meter.summary() for meter in self.meters]
        logger.info(' '.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


class Monitor(object):
    """
    This is an abstract interface for data loggers

    Train monitors log the progress of the training process to some backend.
    This backend can be a file, a web service, or some other means to collect and/or
    display the training
    """

    def __init__(self):
        pass

    def update(self, epoch, step_idx, step_num, prefix, meter_dict):
        raise NotImplementedError


class ProgressMonitor(Monitor):
    def __init__(self, logger):
        super().__init__()
        self.logger = logger

    def update(self, epoch, step_idx, step_num, prefix, meter_dict):
        msg = prefix
        if epoch > -1:
            msg += ' [%d][%5d/%5d]   ' % (epoch, step_idx, int(step_num))
        else:
            msg += ' [%5d/%5d]   ' % (step_idx, int(step_num))
        for k, v in meter_dict.items():
            msg += k + ' '
            if isinstance(v, AverageMeter):
                msg += str(v)
            else:
                msg += '%.6f' % v
            msg += '   '
        self.logger.info(msg)


class TensorBoardMonitor(Monitor):
    def __init__(self, log_dir):
        super().__init__()
        self.writer = SummaryWriter(log_dir / 'tb_runs')
        # logger.info('TensorBoard data directory: %s/tb_runs' % log_dir)

    # def update(self, epoch, step_idx, step_num, prefix, meter_dict):
    #     current_step = epoch * step_num + step_idx
    #     for k, v in meter_dict.items():
    #         val = v.val if isinstance(v, AverageMeter) else v
    #         self.writer.add_scalar(prefix + '/' + k, val, current_step)

    def update(self, epoch, step_idx, step_num, prefix, meters):
        current_step = epoch * step_num + step_idx
        for meter in meters:
            self.writer.add_scalar(prefix+'/'+meter.name, meter.val, current_step)

