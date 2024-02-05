# !/usr/bin/env python
# -*- coding:utf-8 -*-

import math
import torch
import torch.nn as nn


def reduce_grad_scale(dim, grad_scale):
    if dim == 2:
        reduced_grad_scale = torch.sum(grad_scale, dim=1, keepdim=True)
    elif dim == 3:
        reduced_grad_scale = torch.sum(grad_scale, dim=(0, 1), keepdim=True)
    elif dim == 4:
        reduced_grad_scale = torch.sum(grad_scale, dim=(2, 3), keepdim=True)
    else:
        raise ValueError(f'Invalid tensor dim {scaled_input.dim()}')

    return reduced_grad_scale


class LsqQuanGrad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, scale, lb, ub, per_channel):
        scaled_input = input / scale
        output = torch.clamp(scaled_input, min=lb, max=ub)
        output = torch.round(output)
        ctx.lb = lb
        ctx.ub = ub
        ctx.per_channel = per_channel
        ctx.save_for_backward(scaled_input, output)

        return output * scale

    @staticmethod
    def backward(ctx, grad_output):
        scaled_input, round_input = ctx.saved_tensors

        grad_input = grad_output.clone()
        grad_input[scaled_input < ctx.lb] = 0
        grad_input[scaled_input > ctx.ub] = 0
        # grad_input[scaled_input < ctx.lb] *= 0.1
        # grad_input[scaled_input > ctx.ub] *= 0.1
        # upper_bound = scaled_input.max() * 0.9
        # lower_bound = scaled_input.min() * 0.9
        # upper_bound = torch.topk(scaled_input.view(-1),
        #                          int(scaled_input.numel()*0.1))[0][-1]
        # lower_bound = torch.topk(scaled_input.view(-1),
        #                          int(scaled_input.numel()*0.1),
        #                          largest=False)[0][-1]
        # grad_input[scaled_input > upper_bound] = 0
        # grad_input[scaled_input < lower_bound] = 0
        # grad_input = grad_output * (1 / torch.cosh(scaled_input) ** 2)

        grad_scale = round_input - scaled_input
        # grad_scale = round_input.clone()
        # grad_scale = torch.sign(scaled_input) * torch.abs(grad_scale)
        grad_scale[scaled_input < ctx.lb] = ctx.lb
        grad_scale[scaled_input > ctx.ub] = ctx.ub
        grad_scale = grad_scale * grad_output
        if ctx.per_channel:
            grad_scale = reduce_grad_scale(scaled_input.dim(), grad_scale)
        else:
            grad_scale = torch.sum(grad_scale)

        return grad_input, grad_scale, None, None, None


class LsqQuanTanGrad(LsqQuanGrad):
    @staticmethod
    def backward(ctx, grad_output):
        scaled_input, round_inp = ctx.saved_tensors

        grad_input = grad_output.clone()
        grad_input[scaled_input < ctx.lb] = 0
        grad_input[scaled_input > ctx.ub] = 0

        degree_inp = torch.clamp(scaled_input, min=ctx.lb, max=ctx.ub)
        offset_inp = round_inp - degree_inp
        degree_inp = torch.clamp(offset_inp + 0.5, min=1e-5, max=1-1e-5) * torch.pi
        partial_dt = torch.cos(degree_inp) / torch.sin(degree_inp)
        partial_dt[scaled_input < ctx.lb] = ctx.lb
        partial_dt[scaled_input > ctx.ub] = ctx.ub
        grad_scale = partial_dt * grad_output
        if ctx.per_channel:
            grad_scale = reduce_grad_scale(scaled_input.dim(), grad_scale)
        else:
            grad_scale = torch.sum(grad_scale)

        return grad_input, grad_scale, None, None, None


class LsqQuanTanhGrad(LsqQuanGrad):
    @staticmethod
    def backward(ctx, grad_output):
        scaled_input, round_inp = ctx.saved_tensors

        grad_input = grad_output.clone()
        grad_input[scaled_input < ctx.lb] = 0
        grad_input[scaled_input > ctx.ub] = 0

        offset_inp = round_inp - torch.clamp(scaled_input, min=ctx.lb, max=ctx.ub)
        # partial_dt = torch.tanh(offset_inp) * (-0.5 / math.tanh(0.5))
        partial_dt = torch.tanh(3*offset_inp) * (-0.5 / math.tanh(1.5))
        partial_dt[scaled_input < ctx.lb] = ctx.lb
        partial_dt[scaled_input > ctx.ub] = ctx.ub
        grad_scale = partial_dt * grad_output
        if ctx.per_channel:
            grad_scale = reduce_grad_scale(scaled_input.dim(), grad_scale)
        else:
            grad_scale = torch.sum(grad_scale)

        return grad_input, grad_scale, None, None, None


class PactQuanGrad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, alpha, bit):
        ctx.save_for_backward(input)
        ctx.alpha = alpha
        ctx.bit = bit
        output = torch.clamp(input, max=alpha.item())
        scale = (2 ** bit - 1) / alpha
        output = torch.round(output * scale) / scale

        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        lower_bound = input < 0
        upper_bound = input > ctx.alpha
        input_range = ~(lower_bound | upper_bound)
        grad_input = grad_output * input_range
        grad_alpha = torch.sum(grad_output * torch.ge(input, ctx.alpha))

        return grad_input, grad_alpha, None


class PactQuanV2Grad(PactQuanGrad):
    @staticmethod
    def backward(ctx, grad_output):
        input,  = ctx.saved_tensors
        lower_bound = input < 0
        upper_bound = input > ctx.alpha
        input_range = ~(lower_bound | upper_bound)
        grad_input = grad_output * input_range
        partial_alpha = torch.round(input * (2 ** ctx.bit - 1) / ctx.alpha) \
                        / (2 ** ctx.bit - 1) - input / ctx.alpha
        partial_alpha[lower_bound] = 0.
        partial_alpha[upper_bound] = 1.
        grad_alpha = torch.sum(grad_output * partial_alpha)

        return grad_input, grad_alpha, None

