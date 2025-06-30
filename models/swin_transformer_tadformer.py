# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------

# This code is referenced from
# Repository: https://github.com/scale-lab/MTLoRA

import torch
from click import prompt
from torch import Tensor
import torch.nn as nn
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from models.lora import TAModuleLinear

from functools import partial

try:
    import os
    import sys

    kernel_path = os.path.abspath(os.path.join('..'))
    sys.path.append(kernel_path)
    from kernels.window_process.window_process import WindowProcess, WindowProcessReverse

except:
    WindowProcess = None
    WindowProcessReverse = None
    print("[Warning] Fused window process have not been installed. Please refer to get_started.md for installation.")


from einops import rearrange as o_rearrange
def rearrange(*args, **kwargs):
    return o_rearrange(*args, **kwargs).contiguous()

def sep_prompt(x, prompt_length):
    prompt = x[:, :prompt_length, :]
    x = x[:, prompt_length:, :]
    return prompt, x

class CompatLinear(nn.Linear):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, input: Tensor, x_tasks: dict = None) -> Tensor:
        return super().forward(input), None


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0., lora=False, tasks=None, tadmtl=None, layer_idx=0,
                 prompt_cfg: dict = None):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.prompt_cfg = prompt_cfg

        self.fc1 = TAModuleLinear(in_features, hidden_features, r=tadmtl.R[layer_idx],
                                lora_shared_scale=tadmtl.SHARED_SCALE[layer_idx],
                                lora_dropout=tadmtl.DROPOUT[layer_idx], tasks=(
                tasks if (lora or tadmtl.INTERMEDIATE_SPECIALIZATION) else None),
                                trainable_scale_shared=tadmtl.TRAINABLE_SCALE_SHARED,
                                shared_mode=tadmtl.SHARED_MODE,
                                taskfilter=tadmtl.DTF,
                                prompt_cfg = prompt_cfg,
                                layer_name='fc1')

        self.act = act_layer()

        self.fc2 = TAModuleLinear(hidden_features, out_features, r=tadmtl.R[layer_idx],
                                lora_shared_scale=tadmtl.SHARED_SCALE[layer_idx],
                                lora_dropout=tadmtl.DROPOUT[layer_idx], tasks=(
                tasks if (lora or tadmtl.INTERMEDIATE_SPECIALIZATION) else None),
                                trainable_scale_shared=tadmtl.TRAINABLE_SCALE_SHARED,
                                shared_mode=tadmtl.SHARED_MODE,
                                taskfilter=tadmtl.DTF,
                                prompt_cfg = prompt_cfg,
                                layer_name='fc2')

        self.tasks = tasks
        self.drop = nn.Dropout(drop)

        self.softmax = nn.Softmax(dim=-1)

        self.ReLU = nn.ReLU()

    def forward(self, x, x_tasks=None,hw_shapes=None, attn_weight=None, PROMPT_FLAG=False):

        x, fc1_lora_tasks = self.fc1(x, x_tasks,hw_shapes=hw_shapes, attn_weight=attn_weight, PROMPT_FLAG=PROMPT_FLAG)
        x = self.act(x)
        x = self.drop(x)
        if fc1_lora_tasks is not None:
            for task in self.tasks:
                fc1_lora_tasks[task] = self.act(fc1_lora_tasks[task])
                fc1_lora_tasks[task] = self.drop(fc1_lora_tasks[task])
        x, fc2_lora_tasks = self.fc2(x, fc1_lora_tasks,hw_shapes=hw_shapes, attn_weight=attn_weight, PROMPT_FLAG=PROMPT_FLAG)
        x = self.drop(x)
        if fc2_lora_tasks is not None:
            for task in self.tasks:
                fc2_lora_tasks[task] = self.drop(fc2_lora_tasks[task])

        return x, fc2_lora_tasks


def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size,
               W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous(
    ).view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size,
                     window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class WindowAttention(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0., lora=False, tasks=None, tadmtl=None, layer_idx=0,
                 prompt_cfg: dict = None):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.prompt_cfg = prompt_cfg
        self.prompt_len = prompt_cfg.PERTASK_LEN * len(tasks)

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - \
            coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(
            1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - \
            1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index",
                             relative_position_index)

        self.qkv = TAModuleLinear(dim, dim * 3, r=tadmtl.R[layer_idx],
                                lora_shared_scale=tadmtl.SHARED_SCALE[layer_idx], lora_dropout=tadmtl.DROPOUT[layer_idx], tasks=None, bias=qkv_bias,
                                trainable_scale_shared=tadmtl.TRAINABLE_SCALE_SHARED, shared_mode=tadmtl.SHARED_MODE,
                                taskfilter=tadmtl.TPC)

        self.attn_drop = nn.Dropout(attn_drop)



        self.proj = TAModuleLinear(dim, dim, r=tadmtl.R[layer_idx],
                                 lora_shared_scale=tadmtl.SHARED_SCALE[layer_idx],
                                 lora_dropout=tadmtl.DROPOUT[layer_idx], tasks=(
            tasks if (lora or tadmtl.INTERMEDIATE_SPECIALIZATION) else None),
                                 trainable_scale_shared=tadmtl.TRAINABLE_SCALE_SHARED,
                                 shared_mode=tadmtl.SHARED_MODE,
                                 taskfilter=tadmtl.DTF,
                                 layer_name='proj',
                                 prompt_cfg = prompt_cfg)

        self.tasks = tasks
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

        self.attn_weight_processing = nn.GELU()

    def forward(self, x, spa_prompts, mask=None,hw_shapes=None, attn_weight_list=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None

        """
        B_, N, C = x.shape

        nW = B_ // spa_prompts.shape[0]
        prompts_len = self.prompt_cfg.PERTASK_LEN * len(self.tasks)
        spa_prompts = spa_prompts[:, None, :, :].expand(-1, nW, -1, -1).clone().reshape(-1, prompts_len, self.dim)

        x = torch.cat([spa_prompts, x], dim=1)
        B_, N, C = x.shape

        qkv, _ = self.qkv(x)
        qkv = qkv.reshape(B_, N, 3, self.num_heads, C //
                          self.num_heads).permute(2, 0, 3, 1, 4)
        # make torchscript happy (cannot use tensor as tuple)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn_weight = (q @ k.transpose(-2, -1))  # (B*nW, nH, N, N), N=nPrompts+Wh*Ww

        attn = attn_weight * self.scale

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww

        #attn = attn + relative_position_bias.unsqueeze(0)
        attn[:, :, self.prompt_len:, self.prompt_len:] = attn[:, :, self.prompt_len:, self.prompt_len:] + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N)  # + mask.unsqueeze(1).unsqueeze(0)
            attn[:, :, :, self.prompt_len:, self.prompt_len:] = attn[:, :, :, self.prompt_len:,self.prompt_len:] + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C) # B*nW, Wh*Ww+T, C

        # reshaping of attn_weight
        attn_weight = attn_weight[:, :, :self.prompt_len, self.prompt_len:] # (B*nW, nH, T, wh*ww)

        ori_attn_weight =attn_weight

        if self.attn_weight_processing is not None:
            proceesed_attn_weight = self.attn_weight_processing(attn_weight)

        # Append attn_weight
        attn_weight_list.append(ori_attn_weight)

        x, x_proj_lora_tasks = self.proj(x,hw_shapes=hw_shapes, attn_weight=proceesed_attn_weight)
        x = self.proj_drop(x)

        spa_prompts, x = sep_prompt(x, self.prompt_len)  # spa_prompts: (B*nW, T, C)
        spa_prompts = spa_prompts.reshape(B_ // nW, nW, prompts_len, C).mean(dim=1)

        if x_proj_lora_tasks is not None:
            for task in self.tasks:
                x_proj_lora_tasks[task] = self.proj_drop(x_proj_lora_tasks[task])
        return x, x_proj_lora_tasks, attn_weight_list, spa_prompts

    def extra_repr(self) -> str:
        return f'dim={self.dim}, window_size={self.window_size}, num_heads={self.num_heads}'

    def flops(self, N):
        # calculate flops for 1 window with token length of N
        flops = 0
        # qkv = self.qkv(x)
        flops += N * self.dim * 3 * self.dim
        # attn = (q @ k.transpose(-2, -1))
        flops += self.num_heads * N * (self.dim // self.num_heads) * N
        #  x = (attn @ v)
        flops += self.num_heads * N * N * (self.dim // self.num_heads)
        # x = self.proj(x)
        flops += N * self.dim * self.dim
        return flops


class SwinTransformerBlock(nn.Module):
    r""" Swin Transformer Block.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resulotion.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
        fused_window_process (bool, optional): If True, use one kernel to fused window shift & window partition for acceleration, similar for the reversed part. Default: False
    """

    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 fused_window_process=False, lora=False, tasks=None, tadmtl=None, layer_idx=0,
                 prompt_cfg: dict = None,
                 LAST_BLOCK_FLAG=False):
        super().__init__()

        self.LAST_BLOCK_FLAG = LAST_BLOCK_FLAG

        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        self.tasks = tasks

        self.prompt_cfg = prompt_cfg
        self.prompts_len = prompt_cfg.PERTASK_LEN * len(tasks)

        self.lora = lora
        if min(self.input_resolution) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop, lora=lora, tasks=tasks, tadmtl=tadmtl, layer_idx=layer_idx,
            prompt_cfg=prompt_cfg)

        self.drop_path = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim,
                       act_layer=act_layer, drop=drop, lora=lora, tasks=tasks, tadmtl=tadmtl, layer_idx=layer_idx,
                       prompt_cfg = prompt_cfg)

        if self.shift_size > 0:
            # calculate attention mask for SW-MSA
            H, W = self.input_resolution
            img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
            h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

            # nW, window_size, window_size, 1
            mask_windows = window_partition(img_mask, self.window_size)
            mask_windows = mask_windows.view(-1,
                                             self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(
                attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)
        self.fused_window_process = fused_window_process

    def forward(self, x, task_prompts, attn_weight_list=None):

        """
        attn_weight_list : attn weight before weight processing (ex before Softmax or GeLU)

        """

        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        # task prompts norm
        ori_task_prompts = task_prompts
        spa_prompts = self.norm1(task_prompts)

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # cyclic shift
        if self.shift_size > 0:
            if not self.fused_window_process:
                shifted_x = torch.roll(
                    x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
                # partition windows
                # nW*B, window_size, window_size, C
                x_windows = window_partition(shifted_x, self.window_size)
            else:
                x_windows = WindowProcess.apply(
                    x, B, H, W, C, -self.shift_size, self.window_size)
        else:
            shifted_x = x
            # partition windows
            # nW*B, window_size, window_size, C
            x_windows = window_partition(shifted_x, self.window_size)

        # nW*B, window_size*window_size, C
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)

        # W-MSA/SW-MSA
        # nW*B, window_size*window_size, C
        attn_windows, attn_windows_lora_tasks, attn_weight_list, spa_prompts = (
            self.attn(x_windows, spa_prompts, mask=self.attn_mask,hw_shapes=(self.window_size,self.window_size), attn_weight_list=attn_weight_list))

        task_prompts = spa_prompts

        # Attn_weight block selection
        attn_weight = attn_weight_list[-1]  # Default : Last Layer

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)

        # reshaping of attn_weight
        # attn_weight has shape (B*nW, nheads, T+wh*ww, T+wh*ww), N=T+HW
        attn_weight = rearrange(attn_weight, '(b nWh nWw) nheads t (Wh Ww) -> b nheads t (nWh Wh) (nWw Ww)', b=B,
                                nWh=H // self.window_size, nWw=W // self.window_size, Wh=self.window_size,
                                Ww=self.window_size)  # Wh: window height; nWh: number of window along height

        # reverse cyclic shift
        if self.shift_size > 0:
            if not self.fused_window_process:
                shifted_x = window_reverse(
                    attn_windows, self.window_size, H, W)  # B H' W' C
                x = torch.roll(shifted_x, shifts=(
                    self.shift_size, self.shift_size), dims=(1, 2))
            else:
                x = WindowProcessReverse.apply(
                    attn_windows, B, H, W, C, self.shift_size, self.window_size)

            attn_weight = torch.roll(attn_weight, shifts=(self.shift_size, self.shift_size), dims=(3, 4))

        else:
            shifted_x = window_reverse(
                attn_windows, self.window_size, H, W)  # B H' W' C
            x = shifted_x

        if attn_windows_lora_tasks is not None:

            for task in self.tasks:
                self.attn.proj.task_skip_feature[task] = self.attn.proj.task_skip_feature[task].view(-1, self.window_size, self.window_size, C)
                self.attn.proj.task_skip_feature[task] = window_reverse(self.attn.proj.task_skip_feature[task], self.window_size, H, W)

                if self.shift_size > 0:
                    self.attn.proj.task_skip_feature[task] = torch.roll(self.attn.proj.task_skip_feature[task], shifts=(self.shift_size, self.shift_size), dims=(1, 2))
                self.attn.proj.task_skip_feature[task] = self.attn.proj.task_skip_feature[task].view(B, H*W, C)


        if attn_windows_lora_tasks is not None:

            for task in self.tasks:
                attn_windows_lora_tasks[task] = attn_windows_lora_tasks[task].view(-1, self.window_size, self.window_size, C)
                attn_windows_lora_tasks[task] = window_reverse(
                    attn_windows_lora_tasks[task], self.window_size, H, W)
                if self.shift_size > 0:
                    attn_windows_lora_tasks[task] = torch.roll(attn_windows_lora_tasks[task], shifts=(
                        self.shift_size, self.shift_size), dims=(1, 2))
                attn_windows_lora_tasks[task] = attn_windows_lora_tasks[task].view(
                    B, H * W, C)
                attn_windows_lora_tasks[task] = shortcut + \
                    self.drop_path(attn_windows_lora_tasks[task])
        x = x.view(B, H * W, C)
        x = shortcut + self.drop_path(x)

        attn_weight_list[-1] = attn_weight

        mlp_result, mlp_lora_tasks = self.mlp(
            self.norm2(x), {task: self.norm2(attn_windows_lora_tasks[task]) for task in self.tasks} if attn_windows_lora_tasks is not None else None,hw_shapes=(H,W),
            attn_weight= attn_weight)

        if not self.LAST_BLOCK_FLAG:
            task_prompts = ori_task_prompts + self.drop_path(self.mlp(self.norm2(task_prompts), PROMPT_FLAG=True)[0])

        if mlp_lora_tasks is None:

            return x + self.drop_path(mlp_result), None, task_prompts, attn_weight_list
        else:
            if attn_windows_lora_tasks is None:
                for task in self.tasks:
                    mlp_lora_tasks[task] = self.drop_path(mlp_lora_tasks[task])
            else:
                for task in self.tasks:
                    mlp_lora_tasks[task] = attn_windows_lora_tasks[task] + \
                        self.drop_path(mlp_lora_tasks[task])
            return x + self.drop_path(mlp_result), mlp_lora_tasks, task_prompts, attn_weight_list



    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, " \
               f"window_size={self.window_size}, shift_size={self.shift_size}, mlp_ratio={self.mlp_ratio}"


    def flops(self):
        flops = 0
        H, W = self.input_resolution
        # norm1
        flops += self.dim * H * W
        # W-MSA/SW-MSA
        nW = H * W / self.window_size / self.window_size
        flops += nW * self.attn.flops(self.window_size * self.window_size)
        # mlp
        flops += 2 * H * W * self.dim * self.dim * self.mlp_ratio
        # norm2
        flops += self.dim * H * W
        return flops


class PatchMerging(nn.Module):
    r""" Patch Merging Layer.

    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm, prompt_cfg=None):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = CompatLinear(4 * dim, 2 * dim, bias=False)

        self.task_prompts_up = nn.Linear(dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)
        self.prompt_cfg = prompt_cfg


    def forward(self, x, task_prompts):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

        x = x.view(B, H, W, C)

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C

        task_prompts = self.task_prompts_up(task_prompts)
        x = self.norm(x)
        x, _ = self.reduction(x)

        return x, task_prompts

    def upsample_prompt(self, prompt_emb):
        prompt_emb = torch.cat(
                (prompt_emb, prompt_emb, prompt_emb, prompt_emb), dim=-1)
        return prompt_emb

    def extra_repr(self) -> str:
        return f"input_resolution={self.input_resolution}, dim={self.dim}"

    def flops(self):
        H, W = self.input_resolution
        flops = H * W * self.dim
        flops += (H // 2) * (W // 2) * 4 * self.dim * 2 * self.dim
        return flops


class BasicLayer(nn.Module):
    """ A basic Swin Transformer layer for one stage.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
        fused_window_process (bool, optional): If True, use one kernel to fused window shift & window partition for acceleration, similar for the reversed part. Default: False
    """

    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False,
                 fused_window_process=False, tasks=None, tadmtl=None, layer_idx=0,
                 prompt_cfg:dict = None,
                 LAST_BLOCK_FLAG=False):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint
        self.tasks = tasks

        self.TPC = tadmtl.TPC

        self.tadmtl = tadmtl

        self.prompt_cfg = prompt_cfg

        # build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(dim=dim, input_resolution=input_resolution,
                                 num_heads=num_heads, window_size=window_size,
                                 shift_size=0 if (
                                         i % 2 == 0) else window_size // 2,
                                 mlp_ratio=mlp_ratio,
                                 qkv_bias=qkv_bias, qk_scale=qk_scale,
                                 drop=drop, attn_drop=attn_drop,
                                 drop_path=drop_path[i] if isinstance(
                                     drop_path, list) else drop_path,
                                 norm_layer=norm_layer,
                                 fused_window_process=fused_window_process,
                                 lora=(i == depth - 1),
                                 tasks=tasks,
                                 tadmtl=tadmtl,
                                 layer_idx=layer_idx,
                                 prompt_cfg=prompt_cfg,
                                 LAST_BLOCK_FLAG=True if i==depth-1 and LAST_BLOCK_FLAG==True else False
                                 )
            for i in range(depth)])

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(
                input_resolution, dim=dim, norm_layer=norm_layer, prompt_cfg=prompt_cfg)
        else:
            self.downsample = None

        self.attn_weight_processing = nn.GELU()
        self.prompt_gate = nn.Parameter(torch.zeros(1))

    def forward(self, x, task_prompts):

        attn_weight_list = []

        for idx, blk in enumerate(self.blocks):
            x, tasks_lora, task_prompts, attn_weight_list = blk(x, task_prompts, attn_weight_list)

        for t_idx, task in enumerate(self.tasks):
            # print(self.blocks[-1].attn.proj.task_skip_feature[task].shape)
            cur_feature = self.blocks[-1].attn.proj.task_skip_feature[task]

            import torch.nn.functional as F

            gate_clipped = F.sigmoid(self.prompt_gate)

            if self.tadmtl.ABLATION.SKIPCONNECTION:
                #tasks_lora[task] = cur_feature * gate_clipped + (1 - gate_clipped) * tasks_lora[task]
                if self.tadmtl.ABLATION.STAGEWISEGATING:
                    tasks_lora[task] = cur_feature * gate_clipped + (1 - gate_clipped) * tasks_lora[task]
                else:
                    tasks_lora[task] = (cur_feature + tasks_lora[task]) * 0.5



        ori_prompts = task_prompts

        if self.downsample is not None:
            x, task_prompts = self.downsample(x, ori_prompts)
            if tasks_lora is not None:
                for task in self.tasks:
                    tasks_lora[task], _ = self.downsample(tasks_lora[task], ori_prompts)

        return x, tasks_lora, task_prompts, attn_weight_list


    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}"

    def flops(self):
        flops = 0
        for blk in self.blocks:
            flops += blk.flops()
        if self.downsample is not None:
            flops += self.downsample.flops()
        return flops


class PatchEmbed(nn.Module):
    r""" Image to Patch Embedding

    Args:
        img_size (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] //
                              patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv2d(in_chans, embed_dim,
                              kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)  # B Ph*Pw C
        if self.norm is not None:
            x = self.norm(x)
        return x

    def flops(self):
        Ho, Wo = self.patches_resolution
        flops = Ho * Wo * self.embed_dim * self.in_chans * \
            (self.patch_size[0] * self.patch_size[1])
        if self.norm is not None:
            flops += Ho * Wo * self.embed_dim
        return flops


class SwinTransformerTADFormer(nn.Module):
    r""" Swin Transformer
        A PyTorch impl of : `Swin Transformer: Hierarchical Vision Transformer using Shifted Windows`  -
          https://arxiv.org/pdf/2103.14030

    Args:
        img_size (int | tuple(int)): Input image size. Default 224
        patch_size (int | tuple(int)): Patch size. Default: 4
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        embed_dim (int): Patch embedding dimension. Default: 96
        depths (tuple(int)): Depth of each Swin Transformer layer.
        num_heads (tuple(int)): Number of attention heads in different layers.
        window_size (int): Window size. Default: 7
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float): Override default qk scale of head_dim ** -0.5 if set. Default: None
        drop_rate (float): Dropout rate. Default: 0
        attn_drop_rate (float): Attention dropout rate. Default: 0
        drop_path_rate (float): Stochastic depth rate. Default: 0.1
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        ape (bool): If True, add absolute position embedding to the patch embedding. Default: False
        patch_norm (bool): If True, add normalization after patch embedding. Default: True
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False
        fused_window_process (bool, optional): If True, use one kernel to fused window shift & window partition for acceleration, similar for the reversed part. Default: False
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, num_classes=1000,
                 embed_dim=96, depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24],
                 window_size=7, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
                 use_checkpoint=False, fused_window_process=False,
                 basic_layer=BasicLayer, tasks=None, tadmtl=None,
                 prompt_cfg: dict = None,
                 ** kwargs):
        super().__init__()
        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.mlp_ratio = mlp_ratio
        self.tasks = tasks
        self.tadmtl = tadmtl

        # prompt config
        self.prompt_cfg = prompt_cfg
        self.prompt_tasklen = prompt_cfg.PERTASK_LEN
        self.prompt_len = prompt_cfg.PERTASK_LEN * len(tasks)

        # Shallow 방식사용
        self.task_prompts = nn.Parameter(torch.ones(self.prompt_len, embed_dim))
        trunc_normal_(self.task_prompts, mean=1., std=1.)

        # Print lora params:
        if tadmtl is not None:
            print("\ntadmtl params:")
            print(tadmtl)

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        # absolute position embedding
        if self.ape:
            self.absolute_pos_embed = nn.Parameter(
                torch.zeros(1, num_patches, embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=.02)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate,
                                                sum(depths))]  # stochastic depth decay rule

        # build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = basic_layer(dim=int(embed_dim * 2 ** i_layer),
                                input_resolution=(patches_resolution[0] // (2 ** i_layer),
                                                  patches_resolution[1] // (2 ** i_layer)),
                                depth=depths[i_layer],
                                num_heads=num_heads[i_layer],
                                window_size=window_size,
                                mlp_ratio=self.mlp_ratio,
                                qkv_bias=qkv_bias, qk_scale=qk_scale,
                                drop=drop_rate, attn_drop=attn_drop_rate,
                                drop_path=dpr[sum(depths[:i_layer]):sum(
                                    depths[:i_layer + 1])],
                                norm_layer=norm_layer,
                                downsample=PatchMerging if (
                                    i_layer < self.num_layers - 1) else None,
                                use_checkpoint=use_checkpoint,
                                fused_window_process=fused_window_process,
                                tasks=tasks,
                                tadmtl=self.tadmtl,
                                layer_idx=i_layer,
                                prompt_cfg = prompt_cfg,
                                LAST_BLOCK_FLAG = True if i_layer==self.num_layers-1 else False)
            self.layers.append(layer)

        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(
            self.num_features, num_classes) if num_classes > 0 else nn.Identity()

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    def forward_features(self, x, return_stages=False, flatten_ft=False):

        x = self.patch_embed(x)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)

        # get task prompts
        task_prompts = self.task_prompts[None].expand(x.shape[0], -1, -1)

        self.attn_weight_list = []

        if return_stages:
            out = []
        i = 0
        for layer in self.layers:
            x, tasks_lora, task_prompts, attn_weight_list = layer(x, task_prompts)
            if tasks_lora is None:
                tasks_lora = {task: x for task in self.tasks}
            if return_stages:
                out.append((x, tasks_lora))
            i = i + 1

            self.attn_weight_list.append(attn_weight_list)

        if return_stages:
            return out
        else:
            if flatten_ft:
                x = self.avgpool(x.transpose(1, 2))  # B C 1
                x = torch.flatten(x, 1)
            return x

    def forward(self, x, return_stages=False, flatten_ft=False):
        x = self.forward_features(x, return_stages, flatten_ft)
        x = self.head(x)
        return x

    def flops(self, images=None, logger=None, detailed=False):
        flops = 0
        flops += self.patch_embed.flops()
        for i, layer in enumerate(self.layers):
            flops += layer.flops()
        flops += self.num_features * \
            self.patches_resolution[0] * \
            self.patches_resolution[1] // (2 ** self.num_layers)
        flops += self.num_features * self.num_classes
        return flops
