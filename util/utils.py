# !/usr/bin/env python
# -*- coding:utf-8 -*-

import torch
import torch.nn as nn

from quan import QuanConv2d, QuanLinear
from quan.quantizer import Quantizer, IdentityQuan

__all__ = ['extract_param', 'extract_qparam', 'extract_quantizer', 'extract_qmodule']


def extract_param(model):
    var = list()

    for name, module in model.named_children():
        if not isinstance(module, Quantizer):
            parameters = [param for name, param in module._parameters.items()
                          if param is not None]
            var.extend(parameters)
        if len(module._modules) > 0:
            var.extend(extract_param(module))

    return var


def extract_qparam(model):
    var = list()

    for name, module in model.named_children():
        if isinstance(module, Quantizer):
            parameters = [param for name, param in module._parameters.items()
                          if param is not None]
            var.extend(parameters)
        if len(module._modules) > 0:
            var.extend(extract_qparam(module))

    return var


# extract all quantizers.
def extract_quantizer(model):
    var = list()

    for name, module in model.named_children():
        if isinstance(module, Quantizer):
            if isinstance(module, IdentityQuan): continue
            var.append(module)
        if len(module._modules) > 0:
            var.extend(extract_quantizer(module))

    return var


# extract all weights to be quantized.
def extract_qmodule(model):
    var = list()

    for name, module in model.named_modules():
        if isinstance(module, QuanConv2d) or isinstance(module, QuanLinear):
            if isinstance(module.quan_w_fn, IdentityQuan): continue
            var.append(module)

    return var

