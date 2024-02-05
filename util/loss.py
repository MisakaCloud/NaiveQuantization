# !/usr/bin/env python
# -*- coding:utf-8 -*-

import torch
import torch.nn as nn

class MultipleLoss(nn.Module):
    def __init__(self, cross_entropy_loss, aux_criterions):
        super(MultipleLoss, self).__init__()

        self.ce_criterion = cross_entropy_loss
        self.aux_criterions = aux_criterions

    def forward(self, outputs, targets):
        ce_loss = self.ce_criterion(outputs, targets)
        for criterion in self.aux_criterions:
            ce_loss += criterion()

        return ce_loss


class L2RegularizationLoss(nn.Module):
    def __init__(self, parameters, coefficient, device):
        super(L2RegularizationLoss, self).__init__()

        self.parameters = parameters
        self.coefficient = coefficient
        self.device = device

    def forward(self):
        l2_loss = torch.zeros(1).to(self.device)
        for param in self.parameters:
            l2_loss += torch.pow(param, 2)
        l2_loss *= self.coefficient

        return l2_loss.sum()


class WeightClusterLoss(nn.Module):
    def __init__(self, modules, coefficient, bit, per_channel, symmetric, device):
        super(WeightClusterLoss, self).__init__()

        self.modules = modules
        self.coefficient = coefficient
        self.bit = bit
        self.per_channel = per_channel
        self.symmetric = symmetric
        self.device = device

        if self.symmetric:
            self.thd_pos = 2 ** (self.bit-1) - 1
            self.thd_neg = -self.thd_pos
        else:
            self.thd_pos = 2 ** (self.bit-1) - 1
            self.thd_neg = -2 ** (self.bit-1)

    def forward(self):
        q_range = torch.arange(self.thd_neg, self.thd_pos+1, dtype=torch.float)
        q_range = q_range.to(self.device)
        wc_loss = torch.zeros(1).to(self.device)
        for module in self.modules:
            weight_param = module.weight
            weight_param = weight_param.view(weight_param.size()[0], -1)
            scale = module.quan_w_fn.s.detach()
            cluster_centers = torch.squeeze(scale * q_range)
            # for c in range(weight_param.size()[0]):
            #     distances = torch.transpose(weight_param[c].unsqueeze(0), 0, 1) \
            #                 - cluster_centers[c]
            #     values, _ = torch.sort(torch.abs(distances))
            #     wc_loss += torch.sum(values[:, 0])
            distances = weight_param.unsqueeze(dim=2) - cluster_centers.unsqueeze(dim=1)
            values, _ = torch.sort(torch.abs(distances), dim=2)
            wc_loss += torch.sum(values)

        return wc_loss.sum() * self.coefficient


class SoftWeightClusterLoss(WeightClusterLoss):
    def __init__(self, modules, coefficient, bit, per_channel, symmetric, device):
        super(SoftWeightClusterLoss, self).__init__(modules, coefficient, bit,
                                                    per_channel, symmetric, device)

    def forward(self):
        q_range = torch.arange(self.thd_neg, self.thd_pos+1, dtype=torch.float)
        q_range = q_range.to(self.device)
        wc_loss = torch.zeros(1).to(self.device)
        for module in self.modules:
            weight_param = module.weight
            weight_param = weight_param.view(weight_param.size()[0], -1)
            scale = module.quan_w_fn.s.detach()
            cluster_centers = torch.squeeze(scale * q_range)
            # for c in range(weight_param.size()[0]):
            #     distances = torch.transpose(weight_param[c].unsqueeze(0), 0, 1) \
            #                 - cluster_centers[c]
            #     distances = torch.abs(distances)
            #     softmax_dis = torch.softmax(distances, dim=1)
            #     weighted_dis = distances * (1-softmax_dis) / (q_range.numel()-1)
            #     wc_loss += torch.sum(weighted_dis)
            distances = weight_param.unsqueeze(dim=2) - cluster_centers.unsqueeze(dim=1)
            softmax_dis = torch.softmax(torch.abs(distances), dim=2)
            weighted_dis = distances * (1-softmax_dis) / (q_range.numel()-1)
            wc_loss += torch.sum(weighted_dis)

        return wc_loss.sum() * self.coefficient


class TopkWeightClusterLoss(WeightClusterLoss):
    def __init__(self, modules, coefficient, topk, bit, per_channel, symmetric, device):
        super(TopkWeightClusterLoss, self).__init__(modules, coefficient, bit,
                                                    per_channel, symmetric, device)
        self.topk = topk
        assert self.topk <= self.thd_pos - self.thd_neg + 1

    def forward(self):
        q_range = torch.arange(self.thd_neg, self.thd_pos+1, dtype=torch.float)
        q_range = q_range.to(self.device)
        wc_loss = torch.zeros(1).to(self.device)
        for module in self.modules:
            weight_param = module.weight
            weight_param = weight_param.view(weight_param.size()[0], -1)
            scale = module.quan_w_fn.s.detach()
            cluster_centers = torch.squeeze(scale * q_range)
            # for c in range(weight_param.size()[0]):
            #     distances = torch.transpose(weight_param[c].unsqueeze(0), 0, 1) \
            #                 - cluster_centers[c]
            #     values, _ = torch.topk(torch.abs(distances), k=self.topk, dim=1)
            #     softmax_dis = torch.softmax(values, dim=1)
            #     weighted_dis = values * (1-softmax_dis) / self.topk
            #     wc_loss += torch.sum(weighted_dis)
            distances = weight_param.unsqueeze(dim=2) - cluster_centers.unsqueeze(dim=1)
            # values, _ = torch.topk(torch.abs(distances), k=self.topk, dim=2)
            values, _ = torch.topk(torch.pow(distances, 2), k=self.topk, dim=2)
            softmax_val = torch.softmax(values, dim=2)
            weighted_val = values * (1-softmax_val) / (self.topk-1)
            wc_loss += torch.sum(weighted_val)

        return wc_loss.sum() * self.coefficient


class BalancedWeightClusterLoss(WeightClusterLoss):
    def __init__(self, modules, coefficient, percentile, bit, per_channel,
                 symmetric, device):
        super(BalancedWeightClusterLoss, self).__init__(modules, coefficient, bit,
                                                        per_channel, symmetric, device)
        self.percentile = percentile
        if self.percentile == 0.68:
            self.std_dev_num = 1
        elif self.percentile == 0.95:
            self.std_dev_num = 2
        elif self.percentile == 0.997:
            self.std_dev_num = 3
        else:
            raise ValueError('Invalid percentile.')

    def forward(self):
        q_range = torch.arange(self.thd_neg, self.thd_pos+1, dtype=torch.float)
        q_range = q_range.to(self.device)
        wc_loss = torch.zeros(1).to(self.device)
        for module in self.modules:
            weight_param = module.weight
            weight_param = weight_param.view(weight_param.size()[0], -1)
            scale = module.quan_w_fn.s.detach()
            cluster_centers = torch.squeeze(scale * q_range)
            _weight_param = weight_param.detach()
            weight_param_mean = torch.mean(_weight_param, dim=1)
            weight_param_std = torch.std(_weight_param, dim=1)
            for c in range(weight_param.size()[0]):
                lower_bound = weight_param_mean[c] - \
                              weight_param_std[c] * self.std_dev_num
                upper_bound = weight_param_mean[c] + \
                              weight_param_std[c] * self.std_dev_num
                range_step = 2 * self.std_dev_num * \
                             weight_param_std[c] / q_range.numel()
                center_idx = (_weight_param[c] - lower_bound) / range_step
                center_idx = torch.clamp(center_idx, min=0,
                                         max=q_range.numel()-1).type(torch.int64)
                target_centers = cluster_centers[c][center_idx]
                distances = weight_param[c] - target_centers
                # distances = torch.clamp(weight_param[c], min=lower_bound,
                #                         max=upper_bound) - target_centers
                wc_loss += torch.sum(torch.abs(distances))

        return wc_loss.sum() * self.coefficient

