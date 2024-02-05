# !/usr/bin/env python
# -*- coding:utf-8 -*-

import torch
import torch.nn as nn
from torch.autograd import Function

from .quantizer import Quantizer
from ..aux_module import log_exp_clamp, square_clamp
from ..kd_losses import SoftTarget, Logits

def grad_scale(x, scale):
    y = x
    y_grad = x * scale
    return (y - y_grad).detach() + y_grad


def round_pass(x, max_value):
    y = x.round()
    y[y==max_value] -= 1e-6
    y_grad = x
    return (y - y_grad).detach() + y_grad


class AlphaDequantFn(Function):
    @staticmethod
    def forward(ctx, input, alpha, x, x_q):
        ctx.save_for_backward(x, x_q)
        ctx.alpha = alpha

        return input * alpha

    @staticmethod
    def backward(ctx, grad_output):
        x, x_q = ctx.saved_tensors
        grad_input = grad_output * ctx.alpha
        inner_grad = x_q - x
        outer_grad = 1
        # outer_grad = torch.ones_like(x)
        # grad_alpha = torch.sum(inner_grad * (x < 1) +
        #                        outer_grad * (x >= 1))
        grad_alpha = torch.sum((inner_grad * (x < 1) +
                                outer_grad * (x >= 1)) * grad_output)

        return grad_input, grad_alpha, None, None


class NormalizeFn(Function):
    @staticmethod
    def forward(ctx, input, alpha):
        ctx.save_for_backward(input)
        output = torch.clamp(input, -alpha, alpha)
        output = torch.abs(output / alpha)

        return output

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()

        return grad_input


class CompandFn(Function):
    @staticmethod
    def forward(ctx, input, theta, interval_num, quantize_scale):
        # normalized_theta = torch.softmax(theta, dim=0)
        normalized_theta = theta
        # d_i i ∈ [0, interval_num]
        interval_upper = torch.arange(1, interval_num+1, dtype=torch.float,
                                      device=theta.device) / interval_num
        interval_lower = torch.zeros(interval_num, device=theta.device)
        interval_lower[1:] = interval_upper[:-1]
        # beta_i i ∈ [0, interval_num]
        matrix_mask = torch.tril(torch.ones(interval_num, interval_num,
                                            device=theta.device))
        level_upper = torch.mv(matrix_mask, normalized_theta)
        level_lower = torch.zeros(interval_num, device=theta.device)
        level_lower[1:] = level_upper[:-1]
        # slope_i i ∈ [0, interval_num]
        slope = normalized_theta * interval_num

        x_usq = input.unsqueeze(dim=-1)
        compact_out = (x_usq - interval_upper) * slope + level_upper
        compact_ind = (x_usq >= interval_lower) & (x_usq < interval_upper)
        compact_out = torch.sum(compact_out * compact_ind, dim=-1)
        quantize_out = torch.round(compact_out * quantize_scale) / quantize_scale
        x_inv = quantize_out.unsqueeze(dim=-1)
        expand_out = (x_inv - level_upper) / slope + interval_upper
        expand_ind = (x_inv >= level_lower) & (x_inv < level_upper)
        expand_out = torch.sum(expand_out * expand_ind, dim=-1)

        ctx.save_for_backward(input, interval_upper, level_upper, slope,
                              quantize_out, compact_ind, expand_ind)
        ctx.interval_num = interval_num

        return expand_out

    @staticmethod
    def backward(ctx, grad_output):
        input, interval_upper, level_upper, slope = ctx.saved_tensors[:4]
        quantize_out, compact_ind, expand_ind = ctx.saved_tensors[4:]

        grad_level_one = expand_ind / slope
        grad_level_one = grad_level_one.unsqueeze(dim=-1) * compact_ind.unsqueeze(dim=-2)
        grad_level_one = torch.sum(grad_level_one, dim=list(range(grad_level_one.dim()))[:-1])
        grad_level_two = compact_ind.unsqueeze(dim=-1) * expand_ind.unsqueeze(dim=-2)
        grad_level_two = torch.sum(grad_level_two, dim=list(range(grad_level_two.dim()))[:-1])
        grad_level_two = grad_level_two / slope

        grad_slope_one = expand_ind / slope
        grad_slope_one = grad_slope_one.unsqueeze(dim=-1) * compact_ind.unsqueeze(dim=-2)
        grad_slope_one = torch.sum(grad_slope_one, dim=grad_slope_one.dim()-2)
        grad_slope_one = (input.unsqueeze(dim=-1)-interval_upper) * grad_slope_one
        grad_slope_one = torch.sum(grad_slope_one,dim=list(range(grad_slope_one.dim()))[:-1])

        grad_slope_two = compact_ind.unsqueeze(dim=-1) * expand_ind.unsqueeze(dim=-2)
        grad_slope_two = torch.sum(grad_slope_two, dim=grad_slope_two.dim()-2)
        grad_slope_two = (quantize_out.unsqueeze(dim=-1)-level_upper) / (slope ** 2) * \
                          grad_slope_two
        grad_slope_two = torch.sum(grad_slope_two,dim=list(range(grad_slope_two.dim()))[:-1])

        grad_level = grad_level_one - grad_level_two
        grad_slope = grad_slope_one - grad_slope_two

        dSlope_dTheta = ctx.interval_num
        dLevel_dTheta = 1.

        grad_input = grad_output.clone()
        grad_theta = grad_level * dLevel_dTheta + grad_slope * dSlope_dTheta

        return grad_input, grad_theta, None, None


class LcqQuan(Quantizer):
    def __init__(self, bit, interval_num=16, all_positive=False,
                 per_channel=True, kd_loss_mode=None, clamp=None):
        super(LcqQuan, self).__init__(bit)
        assert not per_channel
        self.interval_num = interval_num
        self.per_channel = per_channel
        self.scale = torch.nn.Parameter(torch.ones(1))
        # self.alpha = torch.nn.Parameter(torch.tensor(3.))
        # self.theta = torch.nn.Parameter(torch.zeros(interval_num))
        self.theta = torch.nn.Parameter(torch.ones(interval_num))
        # self.sigma = torch.nn.Parameter(torch.zeros(interval_num))
        self.all_positive = all_positive
        if self.all_positive:
            self.thd_neg =  0
            self.thd_pos =  1
            self.quantize_scale = 2 ** bit - 1
        else:
            self.thd_neg = -1
            self.thd_pos =  1
            self.quantize_scale = 2 ** (bit-1) - 1

        self.kd_loss_mode = kd_loss_mode
        if self.kd_loss_mode == 'st':
            self.criterionKD = SoftTarget()
        elif self.kd_loss_mode == 'logits':
            self.criterionKD = Logits()
        elif self.kd_loss_mode == 'at':
            self.criterionKD = AT()

        if clamp.clamp_mode == 'log_exp_clamp':
            self.clamp_func = log_exp_clamp
            self.clamp_temp = clamp.clamp_temp
        elif clamp.clamp_mode == 'square_clamp':
            self.clamp_func = square_clamp
            self.clamp_temp = clamp.clamp_temp
        else:
            self.clamp_func = None

    def init_from_wht(self, x):
        if self.per_channel:
            self.scale = nn.Parameter(
                x.abs().mean(dim=list(range(1, x.dim())), keepdim=True) * 2 / \
                        (self.thd_pos ** 0.5))
        else:
            self.scale = nn.Parameter(x.abs().mean() * 2 / (self.thd_pos ** 0.5))

    def forward(self, x):
        if self.per_channel:
            # linear layer activation/weight
            if x.dim() == 2:
                num_element = x.shape[1]
            # conv2d layer activation (after im2col)
            elif x.dim() == 3:
                num_element = x.shape[0] * x.shape[1]
            # conv2d layer weight
            elif x.dim() == 4:
                num_element = x.shape[0] * x.shape[2] * x.shape[3]
            else:
                raise ValueError(f'Invalid tensor dim {x.dim()}')
        else:
            num_element = x.numel()
        s_grad_scale = 1.0 / ((self.thd_pos * num_element) ** 0.5)
        # s_scale = grad_scale(self.scale, s_grad_scale)
        if self.all_positive:
            s_scale = grad_scale(torch.abs(self.scale), 3*s_grad_scale)
        else:
            s_scale = grad_scale(torch.abs(self.scale), s_grad_scale)

        # clamp x into [0, 1]
        # x_sign = torch.sign(x)
        # x = x / s_scale
        # x = torch.abs(torch.clamp(x, self.thd_neg, self.thd_pos))
        # x = torch.clamp(x, min_val, max_val) / s_scale
        # clamp x into [-1, 1)
        min_val = self.thd_neg * s_scale.detach()
        max_val = self.thd_pos * s_scale.detach()
        if self.clamp_func:
            x = self.clamp_func.apply(x, min_val, max_val-1e-6, self.clamp_temp)
        else:
            x = torch.clamp(x, min_val, max_val-1e-6)
        x = x / s_scale

        # theta_scale = grad_scale(self.theta, 1.0/(num_element ** 0.5))
        # normalized_theta = torch.softmax(theta_scale, dim=0)
        if self.all_positive:
            theta_scale = grad_scale(self.theta, 3.0/(num_element ** 0.5))
            normalized_theta = torch.softmax(theta_scale, dim=0)
            # d_i i ∈ [1, interval_num]
            interval_upper = torch.arange(1, self.interval_num+1, dtype=torch.float,
                                          device=theta_scale.device) / self.interval_num
            interval_lower = torch.zeros(self.interval_num, device=theta_scale.device)
            interval_lower[1:] = interval_upper[:-1]
            # beta_i i ∈ [1, interval_num]
            matrix_mask = torch.tril(torch.ones(self.interval_num, self.interval_num,
                                                device=theta_scale.device))
            level_upper = torch.mv(matrix_mask, normalized_theta.detach())
            level_upper[-1] = 1. # avoid possible accumulate error
            level_lower = torch.zeros(self.interval_num, device=theta_scale.device)
            level_lower[1:] = level_upper[:-1]
        else:
            theta_scale = grad_scale(self.theta, 1.0/(num_element ** 0.5))
            normalized_theta = torch.softmax(theta_scale, dim=0)
            normalized_theta = normalized_theta * 2
            # d_i i ∈ [1, interval_num]
            interval_upper = torch.arange(1, self.interval_num+1, dtype=torch.float,
                                          device=theta_scale.device) - self.interval_num/2
            interval_upper = interval_upper / (self.interval_num / 2)
            interval_lower = torch.zeros(self.interval_num, device=theta_scale.device)
            interval_lower[1:] = interval_upper[:-1]
            interval_lower[0] = -1.
            # beta_i i ∈ [1, interval_num]
            matrix_mask = torch.tril(torch.ones(self.interval_num, self.interval_num,
                                                device=theta_scale.device))
            level_upper = torch.mv(matrix_mask, normalized_theta.detach()) - 1
            level_upper[-1] = 1. # avoid possible accumulate error
            level_lower = torch.zeros(self.interval_num, device=theta_scale.device)
            level_lower[1:] = level_upper[:-1]
            level_lower[0] = -1.
        # slope_i i ∈ [1, interval_num]
        slope = normalized_theta * self.interval_num

        '''
        normalized_sigma = torch.softmax(self.sigma, dim=0)
        normalized_theta = torch.softmax(self.theta, dim=0)
        matrix_mask = torch.tril(torch.ones(self.interval_num, self.interval_num,
                                            device=self.theta.device))
        # d_i i ∈ [1, interval_num]
        interval_upper = torch.mv(matrix_mask, normalized_sigma.detach())
        interval_upper[-1] = 1. # avoid possible accumulate error
        interval_lower = torch.zeros(self.interval_num, device=self.theta.device)
        interval_lower[1:] = interval_upper[:-1]
        # beta_i i ∈ [1, interval_num]
        level_upper = torch.mv(matrix_mask, normalized_theta.detach())
        level_upper[-1] = 1. # avoid possible accumulate error
        level_lower = torch.zeros(self.interval_num, device=self.theta.device)
        level_lower[1:] = level_upper[:-1]
        # slope_i i ∈ [1, interval_num]
        slope = normalized_theta / normalized_sigma
        '''

        try:
            _x = x.unsqueeze(dim=-1)
            # compact_out = (_x - interval_upper) * slope + level_upper
            # compact_ind = (_x >= interval_lower) & (_x < interval_upper)
            # compact_out = torch.sum(compact_out * compact_ind, dim=-1)
            compact_ind = torch.where((_x >= interval_lower) & (_x < interval_upper))
            compact_slp = torch.tile(slope, _x.shape)[compact_ind].view(x.shape)
            compact_itv = torch.tile(interval_upper, _x.shape)[compact_ind].view(x.shape)
            compact_lvl = torch.tile(level_upper, _x.shape)[compact_ind].view(x.shape)
            compact_out = (x - compact_itv) * compact_slp + compact_lvl

            quantize_out = round_pass(compact_out * self.quantize_scale, self.quantize_scale)
            quantize_out = quantize_out / self.quantize_scale

            _x = quantize_out.unsqueeze(dim=-1)
            # expand_out = (_x - level_upper) / slope + interval_upper
            # expand_ind = (_x >= level_lower) & (_x < level_upper)
            # expand_out = torch.sum(expand_out * expand_ind, dim=-1)
            expand_ind = torch.where((_x >= level_lower) & (_x < level_upper))
            expand_slp = torch.tile(slope, _x.shape)[expand_ind].view(x.shape)
            expand_lvl = torch.tile(level_upper, _x.shape)[expand_ind].view(x.shape)
            expand_itv = torch.tile(interval_upper, _x.shape)[expand_ind].view(x.shape)
            expand_out = (quantize_out - expand_lvl) / expand_slp + expand_itv
        except Exception as e:
            import pdb
            pdb.set_trace()
            print('Error occurred! Please handle it...')

        if self.kd_loss_mode:
            self.kd_loss = self.criterionKD(expand_out, x)
        else:
            self.kd_loss = 0.

        # expand_out = x_sign * expand_out * s_scale
        expand_out = expand_out * s_scale

        return expand_out


class LcqExtendQuan(LcqQuan):
    def __init__(self, bit, interval_num=16, all_positive=False,
                 per_channel=True, kd_loss_mode=None, clamp=None):
        super(LcqExtendQuan, self).__init__(bit, interval_num, all_positive,
                                            per_channel, kd_loss_mode, clamp)

    def forward(self, x):
        if self.per_channel:
            # linear layer activation/weight
            if x.dim() == 2:
                num_element = x.shape[1]
            # conv2d layer activation (after im2col)
            elif x.dim() == 3:
                num_element = x.shape[0] * x.shape[1]
            # conv2d layer weight
            elif x.dim() == 4:
                num_element = x.shape[0] * x.shape[2] * x.shape[3]
            else:
                raise ValueError(f'Invalid tensor dim {x.dim()}')
        else:
            num_element = x.numel()
        s_grad_scale = 1.0 / ((self.thd_pos * num_element) ** 0.5)
        # s_scale = grad_scale(self.scale, s_grad_scale)
        s_scale = grad_scale(self.scale, s_grad_scale)

        # clamp x into [-1, 1)
        range_increment = 0.5 / self.quantize_scale * s_scale.detach()
        min_val = self.thd_neg * s_scale.detach() - range_increment
        max_val = self.thd_pos * s_scale.detach() + range_increment
        if self.clamp_func:
            x = self.clamp_func.apply(x, min_val, max_val-1e-6, self.clamp_temp)
        else:
            x = torch.clamp(x, min_val, max_val-1e-6)
        x = x / s_scale

        theta_scale = grad_scale(self.theta, 1.0/(num_element ** 0.5))
        normalized_theta = torch.softmax(theta_scale, dim=0)
        if self.all_positive:
            # d_i i ∈ [1, interval_num]
            interval_upper = torch.arange(1, self.interval_num+1, dtype=torch.float,
                                          device=theta_scale.device) / self.interval_num
            interval_upper[-1] += 0.5 * self.quantize_scale # extend upper bound
            interval_lower = torch.zeros(self.interval_num, device=theta_scale.device)
            interval_lower[1:] = interval_upper[:-1]
            # beta_i i ∈ [1, interval_num]
            matrix_mask = torch.tril(torch.ones(self.interval_num, self.interval_num,
                                                device=theta_scale.device))
            level_upper = torch.mv(matrix_mask, normalized_theta.detach())
            level_upper[-1] = 1. # avoid possible accumulate error
            level_lower = torch.zeros(self.interval_num, device=theta_scale.device)
            level_lower[1:] = level_upper[:-1]
            level_lower[0] -= 0.5 * self.quantize_scale # extend lower bound
        else:
            normalized_theta = normalized_theta * 2
            # d_i i ∈ [1, interval_num]
            interval_upper = torch.arange(1, self.interval_num+1, dtype=torch.float,
                                          device=theta_scale.device) - self.interval_num/2
            interval_upper = interval_upper / (self.interval_num / 2)
            interval_lower = torch.zeros(self.interval_num, device=theta_scale.device)
            interval_lower[1:] = interval_upper[:-1]
            interval_lower[0] = -1.
            # beta_i i ∈ [1, interval_num]
            matrix_mask = torch.tril(torch.ones(self.interval_num, self.interval_num,
                                                device=theta_scale.device))
            level_upper = torch.mv(matrix_mask, normalized_theta.detach()) - 1
            level_upper[-1] = 1. # avoid possible accumulate error
            level_lower = torch.zeros(self.interval_num, device=theta_scale.device)
            level_lower[1:] = level_upper[:-1]
            level_lower[0] = -1.
        # slope_i i ∈ [1, interval_num]
        slope = normalized_theta * self.interval_num

        '''
        normalized_sigma = torch.softmax(self.sigma, dim=0)
        normalized_theta = torch.softmax(self.theta, dim=0)
        matrix_mask = torch.tril(torch.ones(self.interval_num, self.interval_num,
                                            device=self.theta.device))
        # d_i i ∈ [1, interval_num]
        interval_upper = torch.mv(matrix_mask, normalized_sigma.detach())
        interval_upper[-1] = 1. # avoid possible accumulate error
        interval_lower = torch.zeros(self.interval_num, device=self.theta.device)
        interval_lower[1:] = interval_upper[:-1]
        # beta_i i ∈ [1, interval_num]
        level_upper = torch.mv(matrix_mask, normalized_theta.detach())
        level_upper[-1] = 1. # avoid possible accumulate error
        level_lower = torch.zeros(self.interval_num, device=self.theta.device)
        level_lower[1:] = level_upper[:-1]
        # slope_i i ∈ [1, interval_num]
        slope = normalized_theta / normalized_sigma
        '''

        _x = x.unsqueeze(dim=-1)
        # compact_out = (_x - interval_upper) * slope + level_upper
        # compact_ind = (_x >= interval_lower) & (_x < interval_upper)
        # compact_out = torch.sum(compact_out * compact_ind, dim=-1)
        compact_ind = torch.where((_x >= interval_lower) & (_x < interval_upper))
        compact_slp = torch.tile(slope, _x.shape)[compact_ind].view(x.shape)
        compact_itv = torch.tile(interval_upper, _x.shape)[compact_ind].view(x.shape)
        compact_lvl = torch.tile(level_upper, _x.shape)[compact_ind].view(x.shape)
        compact_out = (x - compact_itv) * compact_slp + compact_lvl

        quantize_out = round_pass(compact_out * self.quantize_scale, self.quantize_scale)
        quantize_out = quantize_out / self.quantize_scale

        _x = quantize_out.unsqueeze(dim=-1)
        # expand_out = (_x - level_upper) / slope + interval_upper
        # expand_ind = (_x >= level_lower) & (_x < level_upper)
        # expand_out = torch.sum(expand_out * expand_ind, dim=-1)
        expand_ind = torch.where((_x >= level_lower) & (_x < level_upper))
        expand_slp = torch.tile(slope, _x.shape)[expand_ind].view(x.shape)
        expand_lvl = torch.tile(level_upper, _x.shape)[expand_ind].view(x.shape)
        expand_itv = torch.tile(interval_upper, _x.shape)[expand_ind].view(x.shape)
        expand_out = (quantize_out - expand_lvl) / expand_slp + expand_itv

        if self.kd_loss_mode:
            self.kd_loss = self.criterionKD(expand_out, x)
        else:
            self.kd_loss = 0.

        expand_out = expand_out * s_scale

        return expand_out

if __name__ == '__main__':
    pass

