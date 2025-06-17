# -*- coding: utf-8 -*-
# @Author  : Tuntun Wang
# @File    : Config.py
# @Software: PyCharm
# coding=utf-8

import ml_collections


def get_config():
    config = ml_collections.ConfigDict()
    config.transformer = ml_collections.ConfigDict()
    config.KV_size = 480  # KV_size = Q1 + Q2 + Q3 + Q4
    config.transformer.num_heads = 4
    config.transformer.num_layers = 4
    config.patch_sizes = [16, 8, 4, 2]
    config.base_channel = 32  # base channel of U-Net
    config.n_classes = 1

    config.n_skip = 3
    config.batch_size = 2
    config.n_patches = 256  # int(args.img_size[0] / args.vit_patches_size) * int(args.img_size[1] / args.vit_patches_size)
    config.h = 16  # int(args.img_size[0] / args.vit_patches_size)
    config.w = 16  # int(args.img_size[1] / args.vit_patches_size)

    config.hidden_size = 128

    config.transformer.embeddings_dropout_rate = 0.1
    config.transformer.attention_dropout_rate = 0.1
    config.transformer.dropout_rate = 0

    return config
