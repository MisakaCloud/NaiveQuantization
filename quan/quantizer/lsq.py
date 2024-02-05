import torch
import torch.nn as nn

from .quantizer import Quantizer
from ..aux_module import log_exp_clamp, square_clamp


def clike_round(x):
    return torch.trunc(x + 0.5 * torch.sign(x))


def grad_scale(x, scale):
    y = x
    y_grad = x * scale
    return (y - y_grad).detach() + y_grad


def round_pass(x):
    y = x.round()
    y_grad = x
    return (y - y_grad).detach() + y_grad


class LsqQuan(Quantizer):
    def __init__(self, bit, all_positive=False, symmetric=False,
                 per_channel=True, clamp=None, extend_range=False):
        super(LsqQuan, self).__init__(bit)

        if all_positive:
            assert not symmetric, "Positive quantization cannot be symmetric"
            # unsigned activation is quantized to [0, 2^b-1]
            self.thd_neg = 0
            self.thd_pos = 2 ** bit - 1
        else:
            if symmetric:
                # signed weight/activation is quantized to [-2^(b-1)+1, 2^(b-1)-1]
                self.thd_neg = - 2 ** (bit - 1) + 1
                self.thd_pos = 2 ** (bit - 1) - 1
            else:
                # signed weight/activation is quantized to [-2^(b-1), 2^(b-1)-1]
                self.thd_neg = - 2 ** (bit - 1)
                self.thd_pos = 2 ** (bit - 1) - 1

        self.per_channel = per_channel
        self.s = torch.nn.Parameter(torch.ones(1))

        if clamp.clamp_mode == 'log_exp_clamp':
            self.clamp_func = log_exp_clamp
            self.clamp_temp = clamp.clamp_temp
        elif clamp.clamp_mode == 'square_clamp':
            self.clamp_func = square_clamp
            self.clamp_temp = clamp.clamp_temp
        else:
            self.clamp_func = None
        self.extend_range = extend_range
        self.kd_loss = 0.

    # def init_from(self, x, *args, **kwargs):
    #     if self.per_channel:
    #         self.s = torch.nn.Parameter(
    #             x.detach().abs().mean(dim=list(range(1, x.dim())), keepdim=True) * 2 / (self.thd_pos ** 0.5))
    #     else:
    #         self.s = torch.nn.Parameter(x.detach().abs().mean() * 2 / (self.thd_pos ** 0.5))
    def init_from_wht(self, x):
        if self.per_channel:
            self.s = nn.Parameter(
                x.abs().mean(dim=list(range(1, x.dim())), keepdim=True) * 2 / \
                        (self.thd_pos ** 0.5))
        else:
            self.s = nn.Parameter(x.abs().mean() * 2 / (self.thd_pos ** 0.5))

    def init_from_act(self, act_l1_norm_mean):
        self.s = torch.ones(1)
        self.s = nn.Parameter(self.s.to(act_l1_norm_mean.device))

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
        s_scale = grad_scale(self.s, s_grad_scale)

        x = x / s_scale
        # x = torch.clamp(x, self.thd_neg, self.thd_pos)
        if self.extend_range:
            min_val = self.thd_neg - 0.5 + 1e-6
            max_val = self.thd_pos + 0.5 - 1e-6
        else:
            min_val = self.thd_neg
            max_val = self.thd_pos

        if self.clamp_func:
            x = self.clamp_func.apply(x, min_val, max_val, self.clamp_temp)
        else:
            x = torch.clamp(x, min_val, max_val)

        x = round_pass(x)
        x = x * s_scale

        return x

class LsqQuanBeta(LsqQuan):
    def __init__(self, bit, all_positive=False, symmetric=False,
                 per_channel=True, clamp=None, extend_range=False):
        super(LsqQuanBeta, self).__init__(bit, all_positive, symmetric, per_channel,
                                          clamp, extend_range)

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
        s_scale = grad_scale(self.s, s_grad_scale)

        min_val = self.thd_neg * s_scale.detach()
        max_val = self.thd_pos * s_scale.detach()
        thd_neg = self.thd_neg
        thd_pos = self.thd_pos
        if self.extend_range:
            min_val -= 0.5 * s_scale.detach()
            max_val += 0.5 * s_scale.detach()
            thd_neg = self.thd_neg - 0.5 + 1e-6
            thd_pos = self.thd_pos + 0.5 - 1e-6

        # try:
        #     assert torch.all(min_val <= max_val)
        # except Exception as e:
        #     print('Error occurred! Please handle it...')
        #     import pdb
        #     pdb.set_trace()

        if self.clamp_func:
            x = self.clamp_func.apply(x, min_val, max_val, self.clamp_temp)
        else:
            x = torch.clamp(x, min_val, max_val)

        x = x / s_scale
        x = torch.clamp(x, thd_neg, thd_pos)
        x = round_pass(x)
        x = x * s_scale

        return x

