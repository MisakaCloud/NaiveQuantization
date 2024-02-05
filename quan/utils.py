import logging
import torch
import torch.nn as nn

from .func import *
from .quantizer import *


def quantizer(default_cfg, this_cfg=None):
    target_cfg = dict(default_cfg)
    if this_cfg is not None:
        for k, v in this_cfg.items():
            target_cfg[k] = v

    if target_cfg['bit'] is None:
        q = IdentityQuan
    elif target_cfg['mode'] == 'lsq':
        q = LsqQuan
    elif target_cfg['mode'] == 'lsqbeta':
        q = LsqQuanBeta
    elif target_cfg['mode'] == 'lcq':
        q = LcqQuan
    else:
        raise ValueError('Cannot find quantizer `%s`', target_cfg['mode'])

    target_cfg.pop('mode')
    return q(**target_cfg)


def find_modules_to_quantize(model, quan_scheduler):
    replaced_modules = dict()
    for name, module in model.named_modules():
        if type(module) in QuanModuleMapping.keys():
            if name in quan_scheduler.excepts:
                replaced_modules[name] = QuanModuleMapping[type(module)](
                    module,
                    # quan_w_fn=quantizer(quan_scheduler.weight,
                    #                     quan_scheduler.excepts[name].weight),
                    # quan_a_fn=quantizer(quan_scheduler.act,
                    #                     quan_scheduler.excepts[name].act)
                    quan_w_fn=quantizer(quan_scheduler.weight, {'bit': None}),
                    quan_a_fn=quantizer(quan_scheduler.act, {'bit': None})
                )
            else:
                replaced_modules[name] = QuanModuleMapping[type(module)](
                    module,
                    quan_w_fn=quantizer(quan_scheduler.weight),
                    quan_a_fn=quantizer(quan_scheduler.act)
                )
            # elif type(module) == nn.ReLU:
            #     if quan_scheduler.pact.use_pact:
            #         replaced_modules[name] = QuanModuleMapping[type(module)](
            #             module,
            #             quan_a_fn=PactQuan(bit=quan_scheduler.act.bit,
            #                                    init_alpha=quan_scheduler.pact.alpha,
            #                                    version=quan_scheduler.pact.version)
            #         )
            #     else:
            #         replaced_modules[name] = QuanModuleMapping[type(module)](
            #             module,
            #             quan_a_fn=IdentityQuan()
            #         )
            # else:
            #     replaced_modules[name] = QuanModuleMapping[type(module)](
            #         module,
            #         quan_w_fn=quantizer(quan_scheduler.weight),
            #         quan_a_fn=quantizer(quan_scheduler.act)
            #     )
        elif name in quan_scheduler.excepts:
            logging.warning('Cannot find module %s in the model, skip it' % name)

    return replaced_modules


def replace_module_by_names(model, modules_to_replace):
    def helper(child: torch.nn.Module):
        for n, c in child.named_children():
            if type(c) in QuanModuleMapping.keys():
                for full_name, m in model.named_modules():
                    if c is m:
                        child.add_module(n, modules_to_replace.pop(full_name))
                        break
            else:
                helper(c)

    helper(model)
    return model


def register_hooks(model, config):
    # def _input_shape_hook(module, input, output):
    #     if isinstance(module, torch.nn.Conv2d):
    #         if not config.act.per_channel: return
    #         assert len(input.shape) == 4
    #         assert len(output.shape) == 4
    #         input_im2col_dim = output.shape[2] * output.shape[3]
    #         input_im2col_dim = torch.Tensor(input_im2col_dim)
    #         module.register_buffer('input_per_channel', input_im2col_dim)
    #     elif isinstance(module, torch.nn.Linear):
    #         if not config.act.per_channel: return
    #         assert len(input.shape) == 2
    #         input_dim = torch.Tensor(input.shape[1])
    #         module.register_buffer('input_per_channel', input_dim)
    #     else:
    #         raise TypeError(f'Expected Conv2d or Linear module, but got {type(module)}')

    def _input_statistic_hook(module, input):
        activation = input[0]
        if isinstance(module, torch.nn.Conv2d):
            if config.act.per_channel:
                unfold = nn.Unfold(kernel_size=module.kernel_size,
                                   stride=module.stride,
                                   padding=module.padding,
                                   dilation=module.dilation)
                im2col = unfold(activation)
                l1_norm_mean = im2col.detach().abs().mean(dim=(0, 1), keepdim=True)
            else:
                l1_norm_mean = activation.detach().abs().mean()
        elif isinstance(module, torch.nn.Linear):
            if config.act.per_channel:
                l1_norm_mean = activation.detach().abs().mean(dim=1, keepdim=True)
            else:
                l1_norm_mean = activation.detach().abs().mean()
        elif isinstance(module, torch.nn.ReLU):
            l1_norm_mean = None
        else:
            raise TypeError(f'Expected Conv2d or Linear module, but got {type(module)}')
        module.register_buffer('input_l1_norm_mean', l1_norm_mean)

    hook_handles = list()
    for name, module in model.named_modules():
        if type(module) in QuanModuleMapping.keys():
            hook_handles.append(module.register_forward_pre_hook(_input_statistic_hook))
            # hook_handles.append(module.register_forward_hook(_input_shape_hook))

    return hook_handles


def unregister_hooks(handles):
    for handle in handles:
        handle.remove()

