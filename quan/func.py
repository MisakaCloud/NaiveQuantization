import numpy as np
import torch
import torch.nn as nn

from .quantizer import IdentityQuan

class QuanConv2d(nn.Conv2d):
    def __init__(self, m: nn.Conv2d, quan_w_fn=None, quan_a_fn=None):
        assert type(m) == nn.Conv2d
        super().__init__(m.in_channels, m.out_channels, m.kernel_size,
                         stride=m.stride,
                         padding=m.padding,
                         dilation=m.dilation,
                         groups=m.groups,
                         bias=True if m.bias is not None else False,
                         padding_mode=m.padding_mode)
        self.quan_w_fn = quan_w_fn
        self.quan_a_fn = quan_a_fn

        self.weight = nn.Parameter(m.weight.detach())
        self.quan_w_fn.init_from_wht(m.weight.detach())
        if m.bias is not None:
            self.bias = nn.Parameter(m.bias.detach())
        # assert hasattr(m, 'input_l1_norm_mean')
        # self.quan_a_fn.init_from_act(m.input_l1_norm_mean)
        # self.quan_a_fn.init_from_act(m.weight.detach())

    def forward(self, x):
        quantized_weight = self.quan_w_fn(self.weight)
        # if not isinstance(self.quan_a_fn, IdentityQuan) and self.quan_a_fn.per_channel:
        #     conv2d_params = dict(kernel_size=self.kernel_size, dilation=self.dilation,
        #                          padding=self.padding, stride=self.stride)
        #     unfold = nn.Unfold(**conv2d_params)
        #     im2col = unfold(x)
        #     im2col = self.quan_a_fn(im2col)
        #     conv2d_params['stride'] = conv2d_params['kernel_size']
        #     output_hw = int(np.sqrt(im2col.shape[2])) * conv2d_params['kernel_size'][0]
        #     fold = nn.Fold(output_size=(output_hw, output_hw), **conv2d_params)
        #     quantized_act = fold(im2col)
        #     _tmp_stride = self.stride
        #     self.stride = conv2d_params['stride']
        #     out = self._conv_forward(quantized_act, quantized_weight, self.bias)
        #     self.stride = _tmp_stride
        #     return out
        # else:
        #     quantized_act = self.quan_a_fn(x)
        #     return self._conv_forward(quantized_act, quantized_weight, self.bias)
        quantized_act = self.quan_a_fn(x)
        return self._conv_forward(quantized_act, quantized_weight, self.bias)


class QuanLinear(nn.Linear):
    def __init__(self, m: nn.Linear, quan_w_fn=None, quan_a_fn=None):
        assert type(m) == nn.Linear
        super().__init__(m.in_features, m.out_features,
                         bias=True if m.bias is not None else False)
        self.quan_w_fn = quan_w_fn
        self.quan_a_fn = quan_a_fn

        self.weight = nn.Parameter(m.weight.detach())
        self.quan_w_fn.init_from_wht(m.weight.detach())
        if m.bias is not None:
            self.bias = nn.Parameter(m.bias.detach())
        # assert hasattr(m, 'input_l1_norm_mean')
        # self.quan_a_fn.init_from_act(m.input_l1_norm_mean)
        # self.quan_a_fn.init_from_act(m.weight.detach())

    def forward(self, x):
        quantized_weight = self.quan_w_fn(self.weight)
        quantized_act = self.quan_a_fn(x)
        return nn.functional.linear(quantized_act, quantized_weight, self.bias)


class QuanReLU(nn.ReLU):
    def __init__(self, m: nn.ReLU, quan_w_fn=None, quan_a_fn=None):
        assert type(m) == nn.ReLU
        super(QuanReLU, self).__init__(m.inplace)
        self.quan_a_fn = quan_a_fn

    def forward(self, x):
        x = nn.functional.relu(x)
        q_x = self.quan_a_fn(x)

        return q_x


QuanModuleMapping = {
    nn.Conv2d: QuanConv2d,
    nn.Linear: QuanLinear,
    # nn.ReLU:   QuanReLU,
}
