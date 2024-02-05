# !/usr/bin/env python
# -*- coding:utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F

class log_exp_clamp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, clamp_min, clamp_max, K):
        # ctx.save_for_backward(x, clamp_min, clamp_max)
        # ctx.K = K
        ctx.save_for_backward(x)
        ctx.clamp_min = clamp_min
        ctx.clamp_max = clamp_max
        ctx.K = K

        lower_bound = x < clamp_min
        upper_bound = x > clamp_max
        midst_bound = ~(lower_bound | upper_bound)
        lower_value =  torch.log(1+torch.exp(K*(x-clamp_min))) / K + clamp_min
        upper_value = -torch.log(1+torch.exp(K*(clamp_max-x))) / K + clamp_max

        clipped_x = lower_value * lower_bound.float() + \
                    x * midst_bound.float() + \
                    upper_value * upper_bound.float()

        return clipped_x

    @staticmethod
    def backward(ctx, grad_output):
        # x, clamp_min, clamp_max, = ctx.saved_tensors
        x, = ctx.saved_tensors
        clamp_min = ctx.clamp_min
        clamp_max = ctx.clamp_max

        lower_bound = x < clamp_min
        upper_bound = x > clamp_max
        midst_bound = ~(lower_bound | upper_bound)

        partial_lower_x = torch.exp(ctx.K*(x-clamp_min)) / \
                          (1+torch.exp(ctx.K*(x-clamp_min)))
        partial_upper_x = torch.exp(ctx.K*(clamp_max-x)) / \
                          (1+torch.exp(ctx.K*(clamp_max-x)))

        # partial_clamp_min = 1 / (1+torch.exp(ctx.K*(x-clamp_min)))
        # partial_clamp_max = 1 / (1+torch.exp(ctx.K*(clamp_max-x)))

        grad_input = grad_output * partial_lower_x * lower_bound.float() + \
                     grad_output * partial_upper_x * upper_bound.float() + \
                     grad_output * midst_bound.float()
        # grad_clamp_min = torch.sum(grad_output*partial_clamp_min*lower_bound.float())
        # grad_clamp_max = torch.sum(grad_output*partial_clamp_max*upper_bound.float())

        return grad_input, None, None, None

class square_clamp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, clamp_min, clamp_max, K):
        # ctx.save_for_backward(x, clamp_min, clamp_max)
        # ctx.K = K
        ctx.save_for_backward(x)
        ctx.clamp_min = clamp_min
        ctx.clamp_max = clamp_max
        ctx.K = K

        lower_bound = x < clamp_min
        upper_bound = x > clamp_max
        midst_bound = ~(lower_bound | upper_bound)
        # lower_value
        lower_value = clamp_min * ctx.K * x / \
                      torch.sqrt(1+torch.pow(ctx.K*x, 2))
        # upper value
        upper_value = clamp_max * ctx.K * x / \
                      torch.sqrt(1+torch.pow(ctx.K*x, 2))

        clipped_x = lower_value * lower_bound.float() + \
                    x * midst_bound.float() + \
                    upper_value * upper_bound.float()

        return clipped_x

    @staticmethod
    def backward(ctx, grad_output):
        # x, clamp_min, clamp_max, = ctx.saved_tensors
        x, = ctx.saved_tensors
        clamp_min = ctx.clamp_min
        clamp_max = ctx.clamp_max

        lower_bound = x < clamp_min
        upper_bound = x > clamp_max
        midst_bound = ~(lower_bound | upper_bound)

        # partial lower x
        partial_lower_x = clamp_min * ctx.K / \
                          torch.pow(1+torch.pow(ctx.K*x, 2), 1.5)
        # partial upper x
        partial_upper_x = clamp_max * ctx.K / \
                          torch.pow(1+torch.pow(ctx.K*x, 2), 1.5)

        # partial_clamp_inl = ctx.K * x / torch.pow(1+torch.pow(ctx.K*x, 2), 2)

        grad_input = grad_output * partial_lower_x * lower_bound.float() + \
                     grad_output * partial_upper_x * upper_bound.float() + \
                     grad_output * midst_bound.float()
        # grad_clamp_min = torch.sum(grad_output*partial_clamp_inl*lower_bound.float())
        # grad_clamp_max = torch.sum(grad_output*partial_clamp_inl*upper_bound.float())

        return grad_input, None, None, None

