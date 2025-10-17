# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------

from .swin_transformer import SwinTransformer
from .swin_transformer_v2 import SwinTransformerV2
from .swin_transformer_moe import SwinTransformerMoE
from .swin_mlp import SwinMLP
from .simmim import build_simmim


def build_model(config, is_pretrain=False):
    model_type = config.arch.lower()

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

    if is_pretrain:
        model = build_simmim(config)
        return model

    if model_type == 'swin':
        model = SwinTransformer(img_size=config.img_size,
                                patch_size=config.SWIN['PATCH_SIZE'],
                                in_chans=config.SWIN['IN_CHANS'],
                                num_classes=config.num_classes,
                                embed_dim=config.SWIN['EMBED_DIM'],
                                depths=config.SWIN['DEPTHS'],
                                num_heads=config.SWIN['NUM_HEADS'],
                                window_size=config.SWIN['WINDOW_SIZE'],
                                mlp_ratio=config.SWIN['MLP_RATIO'],
                                qkv_bias=config.SWIN['QKV_BIAS'],
                                qk_scale=config.SWIN['QK_SCALE'],
                                drop_rate=config.DROP_RATE,
                                drop_path_rate=config.DROP_PATH_RATE,
                                ape=config.SWIN['APE'],
                                norm_layer=layernorm,
                                patch_norm=config.SWIN['PATCH_NORM'],
                                use_checkpoint=False,
                                fused_window_process=False)
    elif model_type == 'swinv2':
        model = SwinTransformerV2(img_size=config.img_size,
                                  patch_size=config.SWINV2['PATCH_SIZE'],
                                  in_chans=config.SWINV2['IN_CHANS'],
                                  num_classes=config.num_classes,
                                  embed_dim=config.SWINV2['EMBED_DIM'],
                                  depths=config.SWINV2['DEPTHS'],
                                  num_heads=config.SWINV2['NUM_HEADS'],
                                  window_size=config.SWINV2['WINDOW_SIZE'],
                                  mlp_ratio=config.SWINV2['MLP_RATIO'],
                                  qkv_bias=config.SWINV2['QKV_BIAS'],
                                  drop_rate=config.DROP_RATE,
                                  drop_path_rate=config.DROP_PATH_RATE,
                                  ape=config.SWINV2['APE'],
                                  patch_norm=config.SWINV2['PATCH_NORM'],
                                  use_checkpoint=False,
                                  pretrained_window_sizes=False)
    elif model_type == 'swin_moe':
        model = SwinTransformerMoE(img_size=config.img_size,
                                   patch_size=config.SWIN_MOE['PATCH_SIZE'],
                                   in_chans=config.SWIN_MOE['IN_CHANS'],
                                   num_classes=config.num_classes,
                                   embed_dim=config.SWIN_MOE['EMBED_DIM'],
                                   depths=config.SWIN_MOE['DEPTHS'],
                                   num_heads=config.SWIN_MOE['NUM_HEADS'],
                                   window_size=config.SWIN_MOE['WINDOW_SIZE'],
                                   mlp_ratio=config.SWIN_MOE['MLP_RATIO'],
                                   qkv_bias=config.SWIN_MOE['QKV_BIAS'],
                                   qk_scale=config.SWIN_MOE['QK_SCALE'],
                                   drop_rate=config.DROP_RATE,
                                   drop_path_rate=config.DROP_PATH_RATE,
                                   ape=config.SWIN_MOE['APE'],
                                   patch_norm=config.SWIN_MOE['PATCH_NORM'],
                                   mlp_fc2_bias=config.SWIN_MOE['MLP_FC2_BIAS'],
                                   init_std=config.SWIN_MOE['INIT_STD'],
                                   use_checkpoint=False,
                                   pretrained_window_sizes=False,
                                   moe_blocks=config.SWIN_MOE['MOE_BLOCKS'],
                                   num_local_experts=config.SWIN_MOE['NUM_LOCAL_EXPERTS'],
                                   top_value=config.SWIN_MOE['TOP_VALUE'],
                                   capacity_factor=config.SWIN_MOE['CAPACITY_FACTOR'],
                                   cosine_router=config.SWIN_MOE['COSINE_ROUTER'],
                                   normalize_gate=config.SWIN_MOE['NORMALIZE_GATE'],
                                   use_bpr=config.SWIN_MOE['USE_BPR'],
                                   is_gshard_loss=config.SWIN_MOE['IS_GSHARD_LOSS'],
                                   gate_noise=config.SWIN_MOE['GATE_NOISE'],
                                   cosine_router_dim=config.SWIN_MOE['COSINE_ROUTER_DIM'],
                                   cosine_router_init_t=config.SWIN_MOE['COSINE_ROUTER_INIT_T'],
                                   moe_drop=config.SWIN_MOE['MOE_DROP'],
                                   aux_loss_weight=config.SWIN_MOE['AUX_LOSS_WEIGHT'])
    elif model_type == 'swin_mlp':
        model = SwinMLP(img_size=config.img_size,
                        patch_size=config.SWIN_MLP['PATCH_SIZE'],
                        in_chans=config.SWIN_MLP['IN_CHANS'],
                        num_classes=config.num_classes,
                        embed_dim=config.SWIN_MLP['EMBED_DIM'],
                        depths=config.SWIN_MLP['DEPTHS'],
                        num_heads=config.SWIN_MLP['NUM_HEADS'],
                        window_size=config.SWIN_MLP['WINDOW_SIZE'],
                        mlp_ratio=config.SWIN_MLP['MLP_RATIO'],
                        drop_rate=config.DROP_RATE,
                        drop_path_rate=config.DROP_PATH_RATE,
                        ape=config.SWIN_MLP['APE'],
                        patch_norm=config.SWIN_MLP['PATCH_NORM'],
                        use_checkpoint=False)
    else:
        raise NotImplementedError(f"Unkown model: {model_type}")

    return model
