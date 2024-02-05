# !/usr/bin/env python
# -*- coding:utf-8 -*-

import torch
import torch.nn as nn

from .quantizer import Quantizer
from .quan_grad import PactQuanGrad, PactQuanV2Grad

class PactQuan(Quantizer):
    def __init__(self, bit, init_alpha, version=None):
        super(PactQuan, self).__init__(bit=bit)

        self.pact_alpha = torch.tensor(init_alpha).float()
        self.pact_alpha = nn.Parameter(self.pact_alpha)
        # self.diff_quan_fun = PactQuanGrad
        if version == 'v2':
            self.diff_quan_fun = PactQuanV2Grad
        else:
            self.diff_quan_fun = PactQuanGrad

    def forward(self, x):
        x = self.diff_quan_fun.apply(x, self.pact_alpha, self.bit)

        return x
