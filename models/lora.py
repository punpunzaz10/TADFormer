# Derived from https://github.com/microsoft/LoRA
#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------

# This code is referenced from
# Repository: https://github.com/scale-lab/MTLoRA

# This code is referenced from
# Repository: https://github.com/prismformore/Multi-Task-Transformer

r"""
    Low Ranking Adaptation for LLMs scheme.

             ┌───────────────────┐
             ┆         h         ┆
             └───────────────────┘
                       ▲
                       |
                       +
                    /     \
    ┌─────────────────┐    ╭───────────────╮     Matrix initialization:
    ┆                 ┆     \      B      /      B = 0
    ┆   pretrained    ┆      \    r*d    /       A = N(0, sigma^2)
    ┆    weights      ┆       ╰─────────╯
    ┆                 ┆       |    r    |        r - rank
    ┆   W e R^(d*d)   ┆       | ◀─────▶ |
    ┆                 ┆       ╭─────────╮
    └─────────────────┘      /     A     \
              ▲             /     d*r     \
               \           ╰───────────────╯
                \                ▲
                 \              /
                  \            /
             ┌───────────────────┐
             ┆         x         ┆
             └───────────────────┘

With LoRA (Low Ranking Adaptation: https://arxiv.org/abs/2106.09685) instead of learning weights of size d*d,
we can freeze the pretrained weights and instead learn two matrices of size d*r and r*d (they will store weight updates
for the pretrained weights): the number of parameters in this case will be reduced drastically (depending on the rank of
course) yet after multiplication of matrices d*r and r*d we will get a matrix d*d which we can sum with frozen
pretrained weights and thus fine-tune the model.

The goal of this approach is to move weight updates into a separate matrix which is decomposed with
two matrices of a lower rank.
"""

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Type, Union, Mapping

import torch
import torch.nn as nn
from torch.nn import functional as F
from typing_extensions import Self

from einops import rearrange as o_rearrange
def rearrange(*args, **kwargs):
    return o_rearrange(*args, **kwargs).contiguous()


class LoRALayer(nn.Module):
    def __init__(self, r: int, lora_alpha: int, lora_dropout: float):
        """Store LoRA specific attributes in a class.

        Args:
            r: rank of the weight update matrices. To make sense of using LoRA the rank should be smaller than the rank of
                the weights of the model. The rank can be as low as 1: https://arxiv.org/pdf/2106.09685.pdf (section 7.2)
            lora_alpha: alpha is needed for scaling updates as alpha/r
                "This scaling helps to reduce the need to retune hyperparameters when we vary r"
                https://arxiv.org/pdf/2106.09685.pdf (section 4.1)
            lora_dropout: dropout that is applied on the input in the LoRA branch (before multiplying by matrix A)
        """
        super().__init__()
        assert r >= 0
        self.r = r
        self.lora_alpha = lora_alpha
        # Optional dropout
        if lora_dropout > 0.0:
            self.lora_dropout = nn.Dropout(p=lora_dropout)
        else:
            self.lora_dropout = lambda x: x
        # Mark the weight as unmerged
        self.merged = False


class LoRALinear(LoRALayer):
    # LoRA implemented in a dense layer
    def __init__(
        self,
        # ↓ this part is for pretrained weights
        in_features: int,
        out_features: int,
        # ↓ the remaining part is for LoRA
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        tasks=None,
        **kwargs,
    ):
        """LoRA wrapper around linear class.

        This class has three weight matrices:
            1. Pretrained weights are stored as `self.linear.weight`
            2. LoRA A matrix as `self.lora_A`
            3. LoRA B matrix as `self.lora_B`
        Only LoRA's A and B matrices are updated, pretrained weights stay frozen.

        Args:
            in_features: number of input features of the pretrained weights
            out_features: number of output features of the pretrained weights
            r: rank of the weight update matrices. To make sense of using LoRA the rank should be smaller than the rank of
                the weights of the model. The rank can be as low as 1: https://arxiv.org/pdf/2106.09685.pdf (section 7.2)
            lora_alpha: alpha is needed for scaling updates as alpha/r
                "This scaling helps to reduce the need to retune hyperparameters when we vary r"
                https://arxiv.org/pdf/2106.09685.pdf (section 4.1)
            lora_dropout: dropout that is applied on the input in the LoRA branch (before multiplying by matrix A)
        """
        super().__init__(r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout)
        self.linear = torch.nn.Linear(
            in_features, out_features, **kwargs)

        # Actual trainable parameters
        if r > 0:
            self.lora_A = nn.Parameter(
                self.linear.weight.new_zeros((r, in_features)))
            self.lora_B = nn.Parameter(
                self.linear.weight.new_zeros((out_features, r)))
            self.scaling = self.lora_alpha / self.r
            self.reset_parameters()

    def reset_parameters(self):
        """Reset all the weights, even including pretrained ones."""
        if hasattr(self, "lora_A"):
            # initialize A the same way as the default for nn.Linear and B to zero
            # Wondering why 'a' is equal to math.sqrt(5)?: https://github.com/pytorch/pytorch/issues/15314
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)

    def merge(self):
        """Merges the LoRA weights into the full-rank weights (W = W + delta_W)."""
        if self.r > 0 and not self.merged:
            # Merge the weights and mark it
            self.linear.weight.data += (self.lora_B @
                                        self.lora_A) * self.scaling
            self.merged = True

    def forward(self, x: torch.Tensor):
        # if weights are merged or rank is less or equal to zero (LoRA is disabled) - it's only a regular nn.Linear forward pass;
        # otherwise in addition do the forward pass with LoRA weights and add it's output to the output from pretrained weights
        pretrained = self.linear(x)
        if self.r == 0 or self.merged:
            return pretrained
        lora = (self.lora_dropout(x) @ self.lora_A.transpose(0, 1)
                @ self.lora_B.transpose(0, 1)) * self.scaling
        return pretrained + lora



class NaiveConvFilter(nn.Module):
    def __init__(self, in_channels, kernel_size, padding):
        super().__init__()
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.padding = padding
        self.conv2d = nn.Conv2d(in_channels, in_channels, kernel_size, padding=padding, groups=in_channels)

    def forward(self, x):
        b, c, h, w = x.shape
        x = self.conv2d(x)
        x = x.view(b, c, x.shape[-2], x.shape[-1])
        return x



class DTF(nn.Module):
    def __init__(self, dim, kernel_size, stride=1, padding=1, groups=1, prompt_cfg=None):
        super().__init__()
        self.dim = dim
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.groups = groups
        self.prompt_cfg = prompt_cfg

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.conv = nn.Conv2d(dim, dim * kernel_size * kernel_size, 1)
        self.chan_FilterNorm = FilterNorm(dim, kernel_size, 'channel', nonlinearity='relu', running_std=True,
                                          running_mean=True)

    def forward(self, x):

        b, c, h, w = x.shape
        weight = self.conv(self.pool(x))
        weight = self.chan_FilterNorm(weight)
        weight = weight.view(b * self.dim, 1, self.kernel_size, self.kernel_size)
        x = F.conv2d(x.reshape(1, -1, h, w), weight, stride=self.stride, padding=self.padding, groups=b * self.groups)
        x = x.view(b, c, x.shape[-2], x.shape[-1])

        return x


# Code refer from DDFN
# https://github.com/theFoxofSky/ddfnet

from torch.nn.init import calculate_gain
class FilterNorm(nn.Module):
    def __init__(self, in_channels, kernel_size, filter_type,
                 nonlinearity='linear', running_std=False, running_mean=False):
        assert filter_type in ('spatial', 'channel', 'new')
        assert in_channels >= 1
        super(FilterNorm, self).__init__()
        self.in_channels = in_channels
        self.filter_type = filter_type
        self.runing_std = running_std
        self.runing_mean = running_mean
        std = calculate_gain(nonlinearity) / kernel_size
        if running_std:
            self.std = nn.Parameter(
                torch.randn(in_channels * kernel_size ** 2) * std, requires_grad=True)
        else:
            self.std = std
        if running_mean:
            self.mean = nn.Parameter(
                torch.randn(in_channels * kernel_size ** 2), requires_grad=True)

    def forward(self, x):
        if self.filter_type == 'spatial':
            b, _, h, w = x.size()
            x = x.reshape(b, self.in_channels, -1, h, w)
            x = x - x.mean(dim=2).reshape(b, self.in_channels, 1, h, w)
            x = x / (x.std(dim=2).reshape(b, self.in_channels, 1, h, w) + 1e-10)
            x = x.reshape(b, _, h, w)
            if self.runing_std:
                x = x * self.std[None, :, None, None]
            else:
                x = x * self.std
            if self.runing_mean:
                x = x + self.mean[None, :, None, None]
        elif self.filter_type == 'channel':
            b = x.size(0)
            c = self.in_channels
            x = x.reshape(b, c, -1)
            x = x - x.mean(dim=2).reshape(b, c, 1)
            x = x / (x.std(dim=2).reshape(b, c, 1) + 1e-10)
            x = x.reshape(b, -1)
            if self.runing_std:
                x = x * self.std[None, :]
            else:
                x = x * self.std
            if self.runing_mean:
                x = x + self.mean[None, :]

        elif self.filter_type == "new":
            b, _, h, w = x.size()
            #print(f"x.shape : {x.shape}")
            x = x.reshape(b, self.in_channels, -1, h, w)
            x = x - x.mean(dim=2).reshape(b, self.in_channels, 1, h, w)
            x = x / (x.std(dim=2).reshape(b, self.in_channels, 1, h, w) + 1e-10)
            x = x.reshape(b, _, h, w)
            if self.runing_std:
                x = x * self.std[None, :, None, None]
            else:
                x = x * self.std
            if self.runing_mean:
                x = x + self.mean[None, :, None, None]

        else:
            raise RuntimeError('Unsupported filter type {}'.format(self.filter_type))
        return x



def sep_prompt(x, prompt_length):
    prompt = x[:, :prompt_length, :]
    x = x[:, prompt_length:, :]
    return prompt, x

def sep_tasks_concat_matrix(x_tasks_concat, tasks_length, tasks):
    x_tasks = {}
    for idx, task in enumerate(tasks):
        x_tasks[task] = x_tasks_concat[:, idx*tasks_length:(idx+1)*tasks_length, :]
    return x_tasks

def concat_tasks_dict(x_tasks_dict, dim=1):
    x_tasks_concat = torch.cat(list(x_tasks_dict.values()), dim=dim)
    return x_tasks_concat


class TSModuleLinear(LoRALayer):
    # LoRA implemented in a dense layer
    def __init__(
        self,
        # ↓ this part is for pretrained weights
        in_features: int,
        out_features: int,
        # ↓ the remaining part is for LoRA
        r: Union[int, Mapping[str, int]] = 0,
        lora_shared_scale: float = 1.0,
        lora_task_scale: float = 1.0,
        lora_dropout: float = 0.0,
        tasks=None,
        trainable_scale_shared=False,
        trainable_scale_per_task=False,
        shared_mode: str = 'matrix',

        taskfilter: dict = None,
        layer_name: str = None,
        prompt_cfg=None,

        **kwargs,
    ):
        assert shared_mode in ['matrix', 'matrixv2',
                               'add', 'addition', 'lora_only']
        if shared_mode == 'add':
            shared_mode = 'addition'
        if shared_mode == 'lora_only':
            tasks = None
        has_tasks = tasks is not None
        if not has_tasks:
            if shared_mode not in ['matrix']:
                shared_mode = 'matrix'

        if isinstance(r, int):
            r = {'shared': r}
        super().__init__(
            r=r['shared'], lora_alpha=lora_shared_scale, lora_dropout=lora_dropout)
        self.linear = torch.nn.Linear(
            in_features, out_features, **kwargs)

        self.tasks = tasks
        self.shared_mode = shared_mode

        # filter use
        filter_use = False
        if layer_name == "proj":
            filter_use = taskfilter.PROJ_ENABLED
        elif layer_name == "fc1":
            filter_use = taskfilter.FC1_ENABLED
        elif layer_name == "fc2":
            filter_use = taskfilter.FC2_ENABLED

        self.taskfilter = taskfilter.ENABLED
        self.taskfilter_cfg = taskfilter

        self.layer_name = layer_name

        self.shared_r = r['shared']

        self.prompt_cfg = prompt_cfg

        self.lora_filter = None

        if r['shared'] > 0:
            if has_tasks and taskfilter.ENABLED:

                self.lora_filter = DTF(r['shared'], kernel_size=3, stride=1, padding=1, groups=r['shared'],
                                        prompt_cfg=prompt_cfg)
                #self.lora_filter = nn.ModuleDict({task: NaiveConvFilter(in_channels=r['shared'], kernel_size=3, padding=1) for task in tasks})

            if self.shared_mode == 'addition':
                assert has_tasks
                self.lora_norm = nn.LayerNorm(out_features)
            elif self.shared_mode == 'matrix' or self.shared_mode == 'matrixv2':
                self.lora_shared_A = nn.Parameter(
                    self.linear.weight.new_zeros((r['shared'], in_features)))
                self.lora_shared_B = nn.Parameter(
                    self.linear.weight.new_zeros((out_features, r['shared'])))
            else:
                raise NotImplementedError
            if trainable_scale_shared:
                self.lora_shared_scale = nn.Parameter(
                    torch.FloatTensor([lora_shared_scale]))
            else:
                self.lora_shared_scale = lora_shared_scale
            self.reset_parameters()

    def reset_parameters(self):
        """Reset all the weights, even including pretrained ones."""
        if hasattr(self, "lora_shared_A"):
            # initialize A the same way as the default for nn.Linear and B to zero
            # Wondering why 'a' is equal to math.sqrt(5)?: https://github.com/pytorch/pytorch/issues/15314
            nn.init.kaiming_uniform_(self.lora_shared_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_shared_B)
        if hasattr(self, "lora_tasks_A"):
            for task in self.tasks:
                nn.init.kaiming_uniform_(
                    self.lora_tasks_A[task], a=math.sqrt(5))
                nn.init.zeros_(self.lora_tasks_B[task])

    def merge(self):
        """Merges the LoRA weights into the full-rank weights (W = W + delta_W)."""
        raise NotImplementedError

    def forward(self, x: torch.Tensor, x_tasks: Dict[str, torch.Tensor] = None, hw_shapes: tuple = None):
        """
        PROMPT_FLAG : Prompted patches only input for shared features, for task-specific featuers, there are no prompts
        attn_weight_list
        """
        # TODO: handle merging
        pretrained = self.linear(x)
        if self.r == 0:
            return pretrained, None
        x = self.lora_dropout(x)

        if self.shared_mode == 'matrix':
            lora = (x @ self.lora_shared_A.transpose(0, 1)
                    @ self.lora_shared_B.transpose(0, 1)) * self.lora_shared_scale

            if self.tasks is not None:
                C = x.shape[-1]

                # rearrange x_tasks
                if x_tasks is None:
                    x_tasks = {task: x for task in self.tasks}

                if self.taskfilter:

                    B, N, C = x.size()
                    h, w = hw_shapes
                    assert N == h * w

                    lora_tasks = {}

                    if self.lora_filter is not None:

                        if x_tasks is None:
                            x_tasks_concat = torch.cat([x, x, x, x], dim=1)
                        else:
                            x_tasks_concat = torch.cat(list(x_tasks.values()), dim=1)
                        x_tasks_concat = x_tasks_concat @ self.lora_shared_A.transpose(0, 1)
                        x_tasks = sep_tasks_concat_matrix(x_tasks_concat, tasks_length=N, tasks=self.tasks)

                        for t_idx, task in enumerate(self.tasks):
                            x_ = x_tasks[task]
                            c = x_.size(2)
                            x_ = x_.reshape(B, h, w, c)
                            x_ = x_.permute(0, 3, 1, 2)
                            x_ = self.lora_filter(x_)
                            #x_ = self.lora_filter[task](x_)
                            x_ = x_.permute(0, 2, 3, 1)
                            x_tasks[task] = x_.reshape(B, N, c)

                        x_tasks_concat = torch.cat(list(x_tasks.values()), dim=1)
                        x_tasks_concat = x_tasks_concat @ self.lora_shared_B.transpose(0, 1) * self.lora_shared_scale
                        x_tasks = sep_tasks_concat_matrix(x_tasks_concat, tasks_length=N, tasks=self.tasks)

                        for task in self.tasks:
                            lora_tasks[task] = pretrained + x_tasks[task]

                    else:
                        for t_idx, task in enumerate(self.tasks):
                            x_ = (x if x_tasks is None else x_tasks[task]) @ self.lora_shared_A.transpose(0, 1)
                            lora_tasks[task] = pretrained + x_ @ self.lora_shared_B.transpose(0, 1) * self.lora_shared_scale
                else:
                    lora_tasks = {}
                    for t_idx, task in enumerate(self.tasks):
                        x_ = (x if x_tasks is None else x_tasks[task]) @ self.lora_shared_A.transpose(0, 1)
                        lora_tasks[task] = pretrained + x_ @ self.lora_shared_B.transpose(0,1) * self.lora_shared_scale

            else:
                lora_tasks = None


        elif self.shared_mode == 'matrixv2':
            lora = (x @ self.lora_shared_A.transpose(0, 1)
                    @ self.lora_shared_B.transpose(0, 1)) * self.lora_shared_scale
            lora_tasks = {
                task: pretrained + lora + (
                            (x if x_tasks is None else x_tasks[task]) @ self.lora_tasks_A[task].transpose(
                        0, 1) @ self.lora_tasks_B[task].transpose(0, 1) * self.lora_task_scale[task])
                for task in self.tasks
            } if self.tasks is not None else None
        elif self.shared_mode == 'addition':
            lora_tasks = {
                task: pretrained + ((x if x_tasks is None else x_tasks[task]) @ self.lora_tasks_A[task].transpose(
                    0, 1) @ self.lora_tasks_B[task].transpose(0, 1) * self.lora_task_scale[task])
                for task in self.tasks
            } if self.tasks is not None else None
            lora = self.lora_norm(torch.sum(torch.stack(
                list(lora_tasks.values()), dim=0), dim=0))

        return pretrained + lora, lora_tasks



class TAModuleLinear(LoRALayer):
    # LoRA implemented in a dense layer
    def __init__(
        self,
        # ↓ this part is for pretrained weights
        in_features: int,
        out_features: int,
        # ↓ the remaining part is for LoRA
        r: Union[int, Mapping[str, int]] = 0,
        lora_shared_scale: float = 1.0,
        lora_dropout: float = 0.0,
        tasks=None,
        trainable_scale_shared=False,
        shared_mode: str = 'matrix',

        taskfilter: dict = None,
        layer_name: str = None,
        prompt_cfg = None,

        **kwargs,
    ):
        assert shared_mode in ['matrix', 'matrixv2',
                               'add', 'addition', 'lora_only']
        if shared_mode == 'add':
            shared_mode = 'addition'
        if shared_mode == 'lora_only':
            tasks = None
        has_tasks = tasks is not None
        if not has_tasks:
            if shared_mode not in ['matrix']:
                shared_mode = 'matrix'

        if isinstance(r, int):
            r = {'shared': r}
        super().__init__(
            r=r['shared'], lora_alpha=lora_shared_scale, lora_dropout=lora_dropout)
        self.linear = torch.nn.Linear(
            in_features, out_features, **kwargs)

        self.tasks = tasks
        self.shared_mode = shared_mode
        self.layer_name = layer_name
        self.shared_r = r['shared']
        self.prompt_cfg = prompt_cfg
        self.lora_filter = None
        self.taskfilter = taskfilter

        if r['shared'] > 0:
            if has_tasks :
                #self.lora_filter = DTF(r['shared'], kernel_size=3, stride=1, padding=1, groups=r['shared'], prompt_cfg=prompt_cfg)
                self.lora_filter = nn.ModuleDict({task: NaiveConvFilter(in_channels=r['shared'], kernel_size=3, padding=1) for task in tasks})


            if prompt_cfg is not None :
                if layer_name == "proj" :
                    self.prompt_layernorm = nn.LayerNorm(in_features)
                    self.task_skip_feature = {}

            if self.shared_mode == 'addition':
                assert has_tasks
                self.lora_norm = nn.LayerNorm(out_features)
            elif self.shared_mode == 'matrix' or self.shared_mode == 'matrixv2':

                self.lora_shared_A = nn.Parameter(
                    self.linear.weight.new_zeros((r['shared'], in_features)))
                self.lora_shared_B = nn.Parameter(
                    self.linear.weight.new_zeros((out_features, r['shared'])))

            else:
                raise NotImplementedError
            if trainable_scale_shared:
                self.lora_shared_scale = nn.Parameter(
                    torch.FloatTensor([lora_shared_scale]))
            else:
                self.lora_shared_scale = lora_shared_scale
            self.reset_parameters()

    def reset_parameters(self):
        """Reset all the weights, even including pretrained ones."""
        if hasattr(self, "lora_shared_A"):
            # initialize A the same way as the default for nn.Linear and B to zero
            # Wondering why 'a' is equal to math.sqrt(5)?: https://github.com/pytorch/pytorch/issues/15314
            nn.init.kaiming_uniform_(self.lora_shared_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_shared_B)
        if hasattr(self, "lora_tasks_A"):
            for task in self.tasks:
                nn.init.kaiming_uniform_(
                    self.lora_tasks_A[task], a=math.sqrt(5))
                nn.init.zeros_(self.lora_tasks_B[task])

    def merge(self):
        """Merges the LoRA weights into the full-rank weights (W = W + delta_W)."""
        raise NotImplementedError


    def forward(self, x: torch.Tensor, x_tasks: Dict[str, torch.Tensor] = None,hw_shapes: tuple =None,
                attn_weight=None,
                PROMPT_FLAG=False):
        """
        PROMPT_FLAG : Prompted patches only input for shared features, for task-specific featuers, there are no prompts
        attn_weight_list
        """
        # TODO: handle merging
        pretrained = self.linear(x)
        if self.r == 0:
            return pretrained, None
        x = self.lora_dropout(x)

        if self.shared_mode == 'matrix':
            lora = (x @ self.lora_shared_A.transpose(0, 1)
                    @ self.lora_shared_B.transpose(0, 1)) * self.lora_shared_scale

            if PROMPT_FLAG:
                return pretrained + lora, None

            if self.tasks is not None:
                C = x.shape[-1]

                if x_tasks is None:
                    x_tasks = {task : x for task in self.tasks}

                if self.layer_name == "proj":
                    spa_attn = attn_weight
                    for task in self.tasks:
                        task_prompts, x_tasks[task] = sep_prompt(x_tasks[task], self.prompt_cfg.PERTASK_LEN*len(self.tasks))

                    for t_idx, task in enumerate(self.tasks):
                        cur_attn_weight = spa_attn[:, :, t_idx*self.prompt_cfg.PERTASK_LEN:(t_idx+1)*self.prompt_cfg.PERTASK_LEN, :] # (b, nheads, prompt_len, h*w)
                        bs, nheads = cur_attn_weight.shape[0:2]
                        cur_task_fea = []
                        head_channel_no = C // nheads
                        x_tasks[task] = rearrange(x_tasks[task], "bs n c -> bs c n")

                        for hea in range(nheads):
                            cur_head_attn = cur_attn_weight[:, hea:hea + 1, :, :]
                            cur_head_attn = cur_head_attn.squeeze(1)
                            cur_task_fea.append(cur_head_attn * x_tasks[task][:, head_channel_no * hea:head_channel_no * (hea + 1), :])
                        cur_task_fea = torch.cat(cur_task_fea, dim=1)

                        self.task_skip_feature[task] = cur_task_fea
                        x_tasks[task] = cur_task_fea + x_tasks[task]

                        x_tasks[task] = x_tasks[task].transpose(2, 1)
                        x_tasks[task] = self.prompt_layernorm(x_tasks[task])


                if self.taskfilter :
                    B, N, C = x.size()
                    prompts_len = self.prompt_cfg.PERTASK_LEN*len(self.tasks)
                    if self.layer_name == "proj":
                        N = N - prompts_len
                        h, w = hw_shapes
                    else:
                        h, w = hw_shapes
                    assert N == h * w
                    lora_tasks={}

                    if self.lora_filter is not None:

                        if x_tasks is None:
                            x_tasks_concat = torch.cat([x, x, x, x], dim=1)
                        else:
                            x_tasks_concat = torch.cat(list(x_tasks.values()), dim=1)
                        x_tasks_concat = x_tasks_concat @ self.lora_shared_A.transpose(0, 1)
                        x_tasks = sep_tasks_concat_matrix(x_tasks_concat, tasks_length=N, tasks=self.tasks)

                        for t_idx, task in enumerate(self.tasks):
                            x_ = x_tasks[task]
                            c = x_.size(2)
                            x_ = x_.reshape(B, h, w, c)
                            x_ = x_.permute(0, 3, 1, 2)
                            #x_ = self.lora_filter(x_)
                            x_ = self.lora_filter[task](x_)
                            x_ = x_.permute(0, 2, 3, 1)
                            x_tasks[task] = x_.reshape(B, N, c)

                        x_tasks_concat = torch.cat(list(x_tasks.values()), dim=1)
                        x_tasks_concat = x_tasks_concat @ self.lora_shared_B.transpose(0, 1) * self.lora_shared_scale
                        x_tasks = sep_tasks_concat_matrix(x_tasks_concat, tasks_length=N, tasks=self.tasks)

                        if self.layer_name == "proj":
                            for task in self.tasks:
                                lora_tasks[task] = pretrained[:, prompts_len:, :] + x_tasks[task]
                        else:
                            for task in self.tasks:
                                lora_tasks[task] = pretrained + x_tasks[task]
                    else:
                        if self.layer_name == "proj":
                            for t_idx, task in enumerate(self.tasks):
                                x_ = (x if x_tasks is None else x_tasks[task]) @ self.lora_shared_A.transpose(0, 1)
                                lora_tasks[task] = pretrained[:, prompts_len:, :] + x_ @ self.lora_shared_B.transpose(0, 1) * self.lora_shared_scale
                        else:
                            for t_idx, task in enumerate(self.tasks):
                                x_ = (x if x_tasks is None else x_tasks[task]) @ self.lora_shared_A.transpose(0, 1)
                                lora_tasks[task] = pretrained + x_ @ self.lora_shared_B.transpose(0, 1) * self.lora_shared_scale

                else:
                    lora_tasks = {}

                    prompts_len = self.prompt_cfg.PERTASK_LEN*len(self.tasks)
                    if self.layer_name == "proj":
                        for t_idx, task in enumerate(self.tasks):
                            x_ = (x if x_tasks is None else x_tasks[task]) @ self.lora_shared_A.transpose(0, 1)
                            lora_tasks[task] = pretrained[:, prompts_len:, :] + x_ @ self.lora_shared_B.transpose(0,
                                                                                                                  1) * self.lora_shared_scale
                    else:
                        for t_idx, task in enumerate(self.tasks):
                            x_ = (x if x_tasks is None else x_tasks[task]) @ self.lora_shared_A.transpose(0, 1)
                            lora_tasks[task] = pretrained + x_ @ self.lora_shared_B.transpose(0,
                                                                                              1) * self.lora_shared_scale
            else:
                lora_tasks=None


        elif self.shared_mode == 'matrixv2':
            lora = (x @ self.lora_shared_A.transpose(0, 1)
                    @ self.lora_shared_B.transpose(0, 1)) * self.lora_shared_scale
            lora_tasks = {
                task: pretrained + lora + ((x if x_tasks is None else x_tasks[task]) @ self.lora_tasks_A[task].transpose(
                    0, 1) @ self.lora_tasks_B[task].transpose(0, 1) * self.lora_task_scale[task])
                for task in self.tasks
            } if self.tasks is not None else None
        elif self.shared_mode == 'addition':
            lora_tasks = {
                task: pretrained + ((x if x_tasks is None else x_tasks[task]) @ self.lora_tasks_A[task].transpose(
                    0, 1) @ self.lora_tasks_B[task].transpose(0, 1) * self.lora_task_scale[task])
                for task in self.tasks
            } if self.tasks is not None else None
            lora = self.lora_norm(torch.sum(torch.stack(
                list(lora_tasks.values()), dim=0), dim=0))

        return pretrained + lora, lora_tasks



class LoRAQKVLinear(LoRALinear):
    # LoRA implemented in a dense layer
    def __init__(
        self,
        # ↓ this part is for pretrained weights
        in_features: int,
        out_features: int,
        # ↓ the remaining part is for LoRA
        n_head: int,
        n_query_groups: int,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        enable_lora: Union[bool, Tuple[bool, bool, bool]] = False,
        **kwargs,
    ):
        """LoRA wrapper around linear class that is used for calculation of q, k and v matrices.

        This class has three weight matrices:
            1. Pretrained weights are stored as `self.linear.weight`
            2. LoRA A matrix as `self.lora_A`
            3. LoRA B matrix as `self.lora_B`
        Only LoRA's A and B matrices are updated, pretrained weights stay frozen.

        Args:
            in_features: number of input features of the pretrained weights
            out_features: number of output features of the pretrained weights
            n_head: number of attention heads
            n_query_groups: number of query groups (see diagram in `lit_gpt/config.py`)
            r: rank of the weight update matrices. To make sense of using LoRA the rank should be smaller than the rank of
                the weights of the model. The rank can be as low as 1: https://arxiv.org/pdf/2106.09685.pdf (section 7.2)
            lora_alpha: alpha is needed for scaling updates as alpha/r
                "This scaling helps to reduce the need to retune hyperparameters when we vary r"
                https://arxiv.org/pdf/2106.09685.pdf (section 4.1)
            lora_dropout: dropout that is applied on the input in the LoRA branch (before multiplying by matrix A)
            enable_lora: MergeLinear class is for attention mechanism where qkv are calculated with a single weight matrix. If we
                don't want to apply LoRA we can set it as False. For example if we want to apply LoRA only to `query`
                and `value` but keep `key` without weight updates we should pass `[True, False, True]`
        """
        super(LoRALinear, self).__init__(
            r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout)
        self.linear = torch.nn.Linear(in_features, out_features, **kwargs)
        self.n_head = n_head
        self.n_query_groups = n_query_groups
        if isinstance(enable_lora, bool):
            enable_lora = [enable_lora] * 3
        assert len(enable_lora) == 3
        self.enable_lora = enable_lora

        # Actual trainable parameters
        # To better understand initialization let's imagine that we have such parameters:
        # ⚬ in_features: 128 (embeddings_size)
        # ⚬ out_features: 384 (3 * embedding_size)
        # ⚬ r: 2
        # ⚬ enable_lora: [True, False, True]
        if r > 0 and any(enable_lora):
            self.lora_A = nn.Parameter(self.linear.weight.new_zeros(
                (r * sum(enable_lora), in_features)))  # (4, 128)
            enable_q, enable_k, enable_v = enable_lora
            self.kv_embd_size = self.linear.in_features // (
                n_head // n_query_groups)
            # qkv_shapes will be used to split a tensor with weights correctly
            qkv_shapes = (
                self.linear.in_features * enable_q,
                self.kv_embd_size * enable_k,
                self.kv_embd_size * enable_v,
            )
            self.qkv_shapes = [s for s in qkv_shapes if s]
            self.lora_B = nn.Parameter(self.linear.weight.new_zeros(
                sum(self.qkv_shapes), r))  # (256, 2))
            # Notes about shapes above
            # - self.lora_A has shape (4, 128): 4 because rank is 2 and LoRA is applied only to two matrices;
            # 128 is the input size of the x (embedding size). (4, 128) and not (128, 4) because later on in
            # F.linear function weights are automatically transposed. In addition conv1d requires channels to
            # be before seq length
            # - self.lora_B has shape (256, 2): 256 because LoRA is applied only to two matrices, so the output is
            # 128*2; 2 tells to have two channels per group for group convolution

            # Scaling:
            # This balances the pretrained model`s knowledge and the new task-specific adaptation
            # https://lightning.ai/pages/community/tutorial/lora-llm/
            # So, set alpha to 1.0 to fully add LoRA. If the LoRA seems to have too much effect (i.e., overfitted), set
            # alpha to lower value. If the LoRA seems to have too little effect, set alpha to higher than 1.0. You can
            # tune these values to your needs. This value can be even slightly greater than 1.0!
            # https://github.com/cloneofsimo/lora
            self.scaling = self.lora_alpha / self.r

            # Compute the indices
            # Indices are needed to properly pad weight updates with zeros. If we want to fine-tune queries and values,
            # but not keys, then the weights update should be:
            #
            # [[ΔW,ΔW,ΔW, ..., 0,0,0, ..., ΔW,ΔW,ΔW,],
            #  [....................................],
            #  [ΔW,ΔW,ΔW, ..., 0,0,0, ..., ΔW,ΔW,ΔW,]]
            #      ↑              ↑            ↑
            # ________________________________________
            # | query         | key       | value    |
            # ----------------------------------------
            self.lora_ind = []
            if enable_q:
                self.lora_ind.extend(range(0, self.linear.in_features))
            if enable_k:
                self.lora_ind.extend(
                    range(self.linear.in_features, self.linear.in_features + self.kv_embd_size))
            if enable_v:
                self.lora_ind.extend(
                    range(self.linear.in_features + self.kv_embd_size, self.linear.out_features))
            self.reset_parameters()

    def zero_pad(self, x: torch.Tensor) -> torch.Tensor:
        """Properly pad weight updates with zeros.

        If, based on `self.enable_lora`, we want to fine-tune queries and values, but not keys,
        then the weights update should be:

        [[ΔW,ΔW,ΔW, ..., 0,0,0, ..., ΔW,ΔW,ΔW,],
         [....................................],
         [ΔW,ΔW,ΔW, ..., 0,0,0, ..., ΔW,ΔW,ΔW,]]
            ↑              ↑            ↑
        ________________________________________
        | query         | key       | value    |
        ----------------------------------------

        Args:
            x: tensor with weights update that will be padded with zeros if necessary

        Returns:
            A tensor with weight updates and zeros for deselected q, k or v
        """
        # we need to do zero padding only if LoRA is disabled for one of QKV matrices
        if all(self.enable_lora):
            return x

        # Let's image that:
        # ⚬ input x has shape (64, 64, 256): (batch_size, sequence_length, embeddings_size)
        # ⚬ embeddings_size: 128
        # ⚬ self.linear.out_features: 384 (3 * embeddings_size)
        # ⚬ enable_lora: [True, False, True]
        # Then x has embeddings_size of 256 (2 * 128 as enable_lora only for query and value, not keys) and expected
        # embeddings_size is 384 (self.linear.out_features), so that means that we need to pad from 256 to 384 with zeros, but
        # only for key updates (this is where self.lora_ind comes in handy)
        # Note: double transpose (in the beginning and in the end) is basically a guard for two-dimensional tensors
        # for example when we want to merge/unmerge LoRA weights and pretrained weights
        x = x.transpose(0, 1)
        result = x.new_zeros(
            (*x.shape[:-1], self.linear.out_features))  # (64, 64, 384)
        result = result.view(-1, self.linear.out_features)  # (4096, 384)
        result = result.index_copy(
            1, torch.tensor(
                self.lora_ind, device=result.device), x.reshape(-1, sum(self.qkv_shapes))
        )  # (4096, 256)
        # (64, 64, 384)
        return result.view((*x.shape[:-1], self.linear.out_features)).transpose(0, 1)

    def conv1d(self, input: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        """An extension of the `torch.nn.functional.conv1d` function with a logic specific to grouped queries.

        If the number of heads is equal to the number of query groups - grouped queries are disabled
        (see scheme in `lit_gpt/config.py:Config`). In this case the combined QKV matrix consists of equally sized
        query, key and value parts, which means we can utilize `groups` argument from `conv1d`: with this argument the
        input and weight matrices will be splitted in equally sized parts and applied separately (like having multiple
        conv layers side by side).

        Otherwise QKV matrix consists of unequally sized parts and thus we have to split input and weight matrices manually,
        apply each part of the weight matrix to the corresponding input's part and concatenate the result.

        Args:
            input: input matrix of shape (B, C, T)
            weight: weight matrix of shape (C_output, rank, 1).
                "C_output" is defined as a sum of embedding sizes for each enabled LoRA layer (see init method of the class).

        Returns:
            A tensor with a shape (B, C_output, T)

        """
        if self.n_head == self.n_query_groups:
            # (B, C_output, T)
            return F.conv1d(input, weight, groups=sum(self.enable_lora))

        # Notation:
        # ⚬ N: number of enabled LoRA layers (self.enable_lora)
        # ⚬ C_output': embeddings size for each LoRA layer (not equal in size)
        # ⚬ r: rank of all LoRA layers (equal in size)

        input_splitted = input.chunk(
            sum(self.enable_lora), dim=1)  # N * (B, C // N, T)
        weight_splitted = weight.split(
            self.qkv_shapes)  # N * (C_output', r, 1)
        return torch.cat(
            # (B, C_output', T)
            [F.conv1d(a, b) for a, b in zip(input_splitted, weight_splitted)], dim=1
        )  # (B, C_output, T)

    def merge(self):
        """Merges the LoRA weights into the full-rank weights (W = W + delta_W)."""

        # Let's assume that:
        # ⚬ self.linear.weight.data: (384, 128) or (3 * embedding_size, embedding_size)
        # ⚬ self.lora_A.data: (4, 128)
        # ⚬ self.lora_B.data: (256, 2)
        if self.r > 0 and any(self.enable_lora) and not self.merged:
            delta_w = self.conv1d(
                self.lora_A.data.unsqueeze(0),  # (4, 128) -> (1, 4, 128)
                self.lora_B.data.unsqueeze(-1),  # (256, 2) -> (256, 2, 1)
            ).squeeze(
                0
            )  # (1, 4, 128) @ (256, 2, 1) -> (1, 256, 128) -> (256, 128)
            # W = W + delta_W (merge)
            # (256, 128) after zero_pad (384, 128)
            self.linear.weight.data += self.zero_pad(delta_w * self.scaling)
            self.merged = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Do the forward pass.

        If LoRA's weights are merged with pretrained ones then it's a simple matrix multiplication.
        If not, then multiply pretrained weights with input, apply LoRA on input and do summation.

        Args:
            x: input tensor of shape (batch_size, context_length, embedding_size)

        Returns:
            Output tensor of shape (batch_size, context_length, 3 * embedding_size)
        """

        # Let's assume that:
        # ⚬ x: (64, 64, 128) or (batch_size, context_length, embedding_size)
        # ⚬ self.linear.weight: (384, 128) or (3 * embedding_size, embedding_size)
        # ⚬ self.lora_A.data: (4, 128)
        # ⚬ self.lora_B.data: (256, 2)

        # if weights are merged or LoRA is disabled (r <= 0 or all `enable_lora` are False) - it's only a regular nn.Linear forward pass;
        # otherwise in addition do the forward pass with LoRA weights and add it's output to the output from pretrained weights
        pretrained = self.linear(x)
        if self.r == 0 or not any(self.enable_lora) or self.merged:
            return pretrained
        # (64, 64, 128) @ (4, 128) -> (64, 64, 4)
        after_A = F.linear(self.lora_dropout(x), self.lora_A)
        # For F.conv1d:
        # ⚬ input: input tensor of shape (mini-batch, in_channels, iW)
        # ⚬ weight: filters of shape (out_channels, in_channels/groups, kW)
        after_B = self.conv1d(
            after_A.transpose(-2, -1),  # (64, 64, 4) -> (64, 4, 64)
            self.lora_B.unsqueeze(-1),  # (256, 2) -> (256, 2, 1)
        ).transpose(
            -2, -1
        )  # (64, 4, 64) @ (256, 2, 1) -> (64, 256, 64) -> (64, 64, 256)
        # (64, 64, 256) after zero_pad (64, 64, 384)
        lora = self.zero_pad(after_B) * self.scaling
        return pretrained + lora


def mark_only_lora_as_trainable(model: nn.Module, bias: str = "none", freeze_patch_embed: bool = False, freeze_norm: bool = False, free_relative_bias: bool = False, freeze_downsample_reduction=False) -> None:
    """Freeze all modules except LoRA's and depending on 'bias' value unfreezes bias weights.

    Args:
        model: model with LoRA layers
        bias:
            ``"none"``: all bias weights will be frozen,
            ``"lora_only"``: only bias weight for LoRA layers will be unfrozen,
            ``"all"``: all bias weights will be unfrozen.

    Raises:
        NotImplementedError: if `bias` not in ["none", "lora_only", "all"]
    """
    def lora_filter(key): return "lora_" in key

    def prompt_filter(key): return "prompt" in key

    def patch_embed_filter(
        key): return not freeze_patch_embed and "patch_embed" in key

    def norm_filter(key): return not freeze_norm and "norm" in key

    def downsample_reduction_filter(
        key): return not freeze_downsample_reduction and "downsample.reduction" in key

    def relative_position_bias_filter(
        key): return not free_relative_bias and "relative_position_bias_table" in key

    def all_filters(key):
        return lora_filter(key) or prompt_filter(key) or patch_embed_filter(key) or norm_filter(key) or downsample_reduction_filter(key) or relative_position_bias_filter(key)

    print(f"LoRA bias mode: {bias}")
    print(f"LoRA Freeze patch_embed: {freeze_patch_embed}")
    print(f"LoRA Freeze norm: {freeze_norm}")
    print(f"LoRA Freeze downsample_reduction: {freeze_downsample_reduction}")
    print(f"LoRA Freeze relative_position_bias: {free_relative_bias}")
    # freeze all layers except LoRA's
    for n, p in model.named_parameters():
        if not all_filters(n):
            p.requires_grad = False

    # depending on the `bias` value unfreeze bias weights
    if bias == "none":
        return
    if bias == "all":
        for n, p in model.named_parameters():
            if "bias" in n:
                p.requires_grad = True
    elif bias == "lora_only":
        for m in model.modules():
            if isinstance(m, LoRALayer) and hasattr(m, "bias") and m.bias is not None:
                m.bias.requires_grad = True
    else:
        raise NotImplementedError


def lora_filter(key: str, value: Any) -> bool:
    return "lora_" in key


def merge_lora_weights(model) -> None:
    """Merge LoRA weights into the full-rank weights to speed up inference."""
    for module in model.modules():
        if isinstance(module, LoRALinear):
            module.merge()


def map_old_state_dict_weights(state_dict: Dict, mapping: Mapping, prefix: str, split_qkv: bool = False) -> Dict:
    unmatched_keys = []
    for checkpoint_name, attribute_name in mapping.items():
        full_checkpoint_name = prefix + checkpoint_name
        if full_checkpoint_name in state_dict:
            full_attribute_name = prefix + attribute_name
            weights = state_dict.pop(
                full_checkpoint_name)
            last_four = ".".join(full_attribute_name.split(".")[-4:])
            if split_qkv and last_four in ["attn.qkv.linear.weight", "attn.qkv.linear.bias"]:
                w_q, w_k, w_v = torch.chunk(weights, chunks=3)
                weight_bias = last_four.split(".")[-1]
                full_attribute_name_without_suffix = ".".join(full_attribute_name.split(".")[
                    :-2])
                state_dict[f"{full_attribute_name_without_suffix}.q.linear.{weight_bias}"] = w_q
                state_dict[f"{full_attribute_name_without_suffix}.k.linear.{weight_bias}"] = w_k
                state_dict[f"{full_attribute_name_without_suffix}.v.linear.{weight_bias}"] = w_v
            else:
                state_dict[full_attribute_name] = weights
        else:
            unmatched_keys.append(checkpoint_name)
    if len(unmatched_keys) > 0:
        print(
            f"WARNING: The following keys from the checkpoint were not mapped: {unmatched_keys}")
    return state_dict
