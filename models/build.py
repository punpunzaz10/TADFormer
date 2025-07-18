# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------

from .swin_transformer import SwinTransformer
from .swin_mtl import MultiTaskSwin
from .swin_transformer_tadformer import SwinTransformerTADFormer


def build_model(config, is_pretrain=False):
    model_type = config.MODEL.TYPE

    # accelerate layernorm
    if config.FUSED_LAYERNORM:
        try:
            import apex as amp
            layernorm = amp.normalization.FusedLayerNorm
        except:
            layernorm = None
            print("To use FusedLayerNorm, please install apex.")
    else:
        import torch.nn as nn
        layernorm = nn.LayerNorm

    if model_type == 'swin':
        if config.MODEL.TADMTL.ENABLED and config.MODEL.TADMTL.DTF.ENABLED:
            print("SwinTransformerPromptMTLoRA")

            model = SwinTransformerTADFormer(img_size=config.DATA.IMG_SIZE,
                                          patch_size=config.MODEL.SWIN.PATCH_SIZE,
                                          in_chans=config.MODEL.SWIN.IN_CHANS,
                                          num_classes=config.MODEL.NUM_CLASSES,
                                          embed_dim=config.MODEL.SWIN.EMBED_DIM,
                                          depths=config.MODEL.SWIN.DEPTHS,
                                          num_heads=config.MODEL.SWIN.NUM_HEADS,
                                          window_size=config.MODEL.SWIN.WINDOW_SIZE,
                                          mlp_ratio=config.MODEL.SWIN.MLP_RATIO,
                                          qkv_bias=config.MODEL.SWIN.QKV_BIAS,
                                          qk_scale=config.MODEL.SWIN.QK_SCALE,
                                          drop_rate=config.MODEL.DROP_RATE,
                                          drop_path_rate=config.MODEL.DROP_PATH_RATE,
                                          ape=config.MODEL.SWIN.APE,
                                          norm_layer=layernorm,
                                          patch_norm=config.MODEL.SWIN.PATCH_NORM,
                                          use_checkpoint=config.TRAIN.USE_CHECKPOINT,
                                          fused_window_process=config.FUSED_WINDOW_PROCESS,
                                          tasks=config.TASKS,
                                          tadmtl=config.MODEL.TADMTL,
                                          prompt_cfg = config.MODEL.TADMTL.TPC.PROMPT_CFG)


        else:
            model = SwinTransformer(img_size=config.DATA.IMG_SIZE,
                                    patch_size=config.MODEL.SWIN.PATCH_SIZE,
                                    in_chans=config.MODEL.SWIN.IN_CHANS,
                                    num_classes=config.MODEL.NUM_CLASSES,
                                    embed_dim=config.MODEL.SWIN.EMBED_DIM,
                                    depths=config.MODEL.SWIN.DEPTHS,
                                    num_heads=config.MODEL.SWIN.NUM_HEADS,
                                    window_size=config.MODEL.SWIN.WINDOW_SIZE,
                                    mlp_ratio=config.MODEL.SWIN.MLP_RATIO,
                                    qkv_bias=config.MODEL.SWIN.QKV_BIAS,
                                    qk_scale=config.MODEL.SWIN.QK_SCALE,
                                    drop_rate=config.MODEL.DROP_RATE,
                                    drop_path_rate=config.MODEL.DROP_PATH_RATE,
                                    ape=config.MODEL.SWIN.APE,
                                    norm_layer=layernorm,
                                    patch_norm=config.MODEL.SWIN.PATCH_NORM,
                                    use_checkpoint=config.TRAIN.USE_CHECKPOINT,
                                    fused_window_process=config.FUSED_WINDOW_PROCESS)
    else:
        raise NotImplementedError(f"Unkown model: {model_type}")

    return model


def build_mtl_model(backbone, config):
    model = MultiTaskSwin(backbone, config)
    return model
