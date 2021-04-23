""" Utilities """
import os
import logging
import shutil
import torch
import numpy as np
from ptflops import get_model_complexity_info

def para_flops_for_ops(model):
    flops, params = get_model_complexity_info(model)
    return flops


def param_size(model):
    """ Compute parameter size in MB """
    n_params = sum(
        np.prod(v.size()) for k, v in model.named_parameters() if not k.startswith('aux_head'))
    return n_params / 1024. / 1024.


def count_parameters(model, trainable_only=True, is_dict=False):
    if is_dict:
        return sum(np.prod(list(model[k].size())) for k in model)
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    else:
        return sum(p.numel() for p in model.parameters())


def get_logger(file_path):
    """ Make python logger """
    # [!] Since tensorboardX use default logger (e.g. logging.info()), we should use custom logger
    logger = logging.getLogger('darts')
    log_format = '%(asctime)s | %(message)s'
    formatter = logging.Formatter(log_format, datefmt='%m/%d %I:%M:%S %p')
    file_handler = logging.FileHandler(file_path)
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    logger.setLevel(logging.INFO)

    return logger


class AverageMeter():
    """ Computes and stores the average and current value """
    def __init__(self):
        self.reset()

    def reset(self):
        """ Reset all statistics """
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        """ Update statistics """
        self.val = list(val.values())[0] if isinstance(val, dict) else val
        self.sum += self.val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """ Computes the precision@k for the specified values of k """
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    # one-hot case
    if target.ndimension() > 1:
        target = target.max(1)[1]

    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(1.0 / batch_size))

    return res


def save_checkpoint(state, ckpt_dir, is_best=False):
    filename = os.path.join(ckpt_dir, 'checkpoint.pth.tar')
    if is_best:
        torch.save(state, filename)
        # best_filename = os.path.join(ckpt_dir, 'best.pth.tar')
        # shutil.copyfile(filename, best_filename)

