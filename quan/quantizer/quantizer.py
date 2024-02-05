# !/usr/bin/env python
# -*- coding:utf-8 -*-

import torch

class Quantizer(torch.nn.Module):
    def __init__(self, bit):
        super().__init__()
        self.bit = bit

    def init_from(self, x, *args, **kwargs):
        pass

    def init_from_wht(self, x):
        pass

    def init_from_act(self, x):
        pass

    def forward(self, x):
        raise NotImplementedError


class IdentityQuan(Quantizer):
    def __init__(self, bit=None, *args, **kwargs):
        super().__init__(bit)
        assert bit is None, 'The bit-width of identity quantizer must be None'

    def forward(self, x):
        return x

