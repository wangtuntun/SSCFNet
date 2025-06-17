# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import  Dropout, Softmax, Linear,LayerNorm

def swish(x):
    return x * torch.sigmoid(x)

ACT2FN = {"gelu": torch.nn.functional.gelu, "relu": torch.nn.functional.relu, "swish": swish}

class Mlp(nn.Module):
    def __init__(self, config):
        super(Mlp, self).__init__()
        # torch.nn.Linear
        self.fc1 = Linear(config.n_patches, config.hidden_size)
        self.fc2 = Linear(config.hidden_size, config.n_patches)
        self.act_fn = ACT2FN["gelu"]
        self.dropout = Dropout(config.transformer["dropout_rate"])

    def forward(self, x):
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class Attention(nn.Module):
    def __init__(self, config):
        super(Attention, self).__init__()
        self.num_attention_heads = config.transformer.num_heads
        self.attention_head_size = int(config.n_patches / self.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = Linear(config.n_patches, config.n_patches)
        self.key = Linear(config.n_patches, config.n_patches)
        self.value = Linear(config.n_patches, config.n_patches)

        self.out = Linear(config.n_patches, config.n_patches)
        tmp = config.transformer["attention_dropout_rate"]  # 0.0
        self.attn_dropout = Dropout(config.transformer["attention_dropout_rate"])
        self.proj_dropout = Dropout(config.transformer["attention_dropout_rate"])

        self.softmax = Softmax(dim=-1)
        self.position_embeddings = nn.Parameter(
            torch.randn(1, self.num_attention_heads, config.n_classes, config.n_classes))

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states):
        x_input=hidden_states
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        # print(mixed_query_layer.shape)
        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        attention_scores = attention_scores + self.position_embeddings  # RPE

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = self.softmax(attention_scores)
        # weights = attention_probs if self.vis else None
        attention_probs = self.attn_dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        attention_output = self.out(context_layer)
        attention_output = self.proj_dropout(attention_output)
        x_output=attention_output
        return attention_output


class Block(nn.Module):
    def __init__(self, config):
        super(Block, self).__init__()

        # torch.nn.LayerNorm
        self.attention_norm = LayerNorm(config.n_patches, eps=1e-6)
        self.ffn_norm = LayerNorm(config.n_patches, eps=1e-6)
        self.ffn = Mlp(config)
        self.attn = Attention(config)

    def forward(self, x):
        x_in=x
        h = x
        x = self.attention_norm(x)
        x = self.attn(x)
        x = x + h

        h = x
        x = self.ffn_norm(x)
        x = self.ffn(x)
        x = x + h
        x_out=x
        return x


class self_update_block(nn.Module):
    def __init__(self, config):
        super(self_update_block, self).__init__()
        num_layers = 2
        self.layer = nn.ModuleList()
        self.encoder_norm = LayerNorm(config.n_patches, eps=1e-6) 
        for _ in range(num_layers):
            layer = Block(config) 
            self.layer.append(copy.deepcopy(layer))

    def forward(self, refined_shape_prior):
        x_input=refined_shape_prior # (2,2,256)
        for layer_block in self.layer:
            refined_shape_prior = layer_block(refined_shape_prior)

        encoded = self.encoder_norm(refined_shape_prior)
        x_output=encoded # (2,2,256)
        return encoded


class cross_update_block(nn.Module):
    def __init__(self, n_class):
        super(cross_update_block, self).__init__()
        self.n_class = n_class
        self.softmax = Softmax(dim=-1)

    def forward(self, refined_shape_prior, feature):
        class_feature = torch.matmul(feature.flatten(2), refined_shape_prior.flatten(2).transpose(-1, -2))
        # scale
        class_feature = class_feature / math.sqrt(self.n_class)
        class_feature = self.softmax(class_feature)
        class_feature = torch.einsum("ijk, ikhw->ijhw", class_feature, refined_shape_prior)
        class_feature = feature + class_feature
        return class_feature

class Conv2dReLU(nn.Sequential): 
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            padding=0,
            stride=1,
            use_batchnorm=True,
    ):
    
        conv=nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=not (use_batchnorm),
        )
        relu = nn.ReLU(inplace=True)

        bn=nn.BatchNorm2d(out_channels)

        super(Conv2dReLU, self).__init__(conv, bn, relu)

class Conv2dbn(nn.Sequential):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            padding=0,
            stride=1,
            use_batchnorm=True,
    ):

        conv=nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=not (use_batchnorm),
        )
        bn = nn.BatchNorm2d(out_channels)

        super(Conv2dbn, self).__init__(conv, bn)


class DecoderResBlock(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            use_batchnorm=True,
    ):
        super().__init__()
    
        self.conv1 = Conv2dReLU(
            in_channels,
            out_channels,
            kernel_size=1,
            padding=0,
            use_batchnorm=use_batchnorm,
        )

        self.conv2 = Conv2dReLU(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )

        self.conv3 = Conv2dbn(
            in_channels,
            out_channels,
            kernel_size=1,
            padding=0,
            use_batchnorm=use_batchnorm,
        )

        self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, skip=None):
        x_input=x # (2,512,32,32)
        feature_in = self.conv3(x)
        x = self.conv1(x)
        x = self.conv2(x)
        # shortcut
        x = x + feature_in
        x = self.relu(x)
        x_out=x
        return x


class SPM(nn.Module):
    def __init__(self, config, in_channel, scale):
        super(SPM, self).__init__()
        self.scale = scale
        self.SUB = self_update_block(config) 
        self.CUB  = cross_update_block(config.n_classes)
        self.resblock1 = DecoderResBlock(in_channel, in_channel) 
        self.resblock2 = DecoderResBlock(in_channel, in_channel)
        self.resblock3 = DecoderResBlock(in_channel, config.n_classes) 

        self.h = config.h
        self.w = config.w 
        self.dim = in_channel
        
        
    def forward(self, feature, refined_shape_prior):
        feature_in=feature 
        shape_in=refined_shape_prior 

        b, n_class,_ = refined_shape_prior.size()  

        refined_shape_prior = self.SUB(refined_shape_prior)
        previous_class_center = refined_shape_prior

        refined_shape_prior = F.interpolate(refined_shape_prior.contiguous().view(b, n_class, self.h, self.w), scale_factor=self.scale, mode="bilinear")

        feature = self.resblock1(feature)
        feature = self.resblock2(feature)

        class_feature = self.CUB(refined_shape_prior, feature)

        tmp=self.resblock3(class_feature)
        refined_shape_prior = F.interpolate(self.resblock3(class_feature), scale_factor=(1.0 / self.scale[0], 1.0 / self.scale[1]), mode="bilinear").flatten(2) + previous_class_center

        feature_out=class_feature
        shape_out=refined_shape_prior
        return class_feature, refined_shape_prior

def get_config():
    import ml_collections
    config = ml_collections.ConfigDict()
    config.transformer = ml_collections.ConfigDict()
    config.KV_size = 480  # KV_size = Q1 + Q2 + Q3 + Q4

    config.transformer.num_heads = 2
    config.transformer.num_layers = 2
    config.patch_sizes = [16, 8, 4, 2]
    config.base_channel = 32  # base channel of U-Net

    config.n_classes = 1

    config.n_skip = 3
    config.batch_size = 2
    config.n_patches = 256 # int(args.img_size[0] / args.vit_patches_size) * int(args.img_size[1] / args.vit_patches_size)
    config.h = 16 #int(args.img_size[0] / args.vit_patches_size)
    config.w = 16 #int(args.img_size[1] / args.vit_patches_size)

    config.hidden_size =128

    config.transformer.embeddings_dropout_rate = 0.1
    config.transformer.attention_dropout_rate = 0.1
    config.transformer.dropout_rate = 0

    return config


def test_spm_config():
    config_vit = get_config()
    spm_input = torch.randn(2, 8, 128, 128)
    learnable_shape_prior = nn.Parameter(torch.randn(2, 256))
    learnable_shape_prior = learnable_shape_prior.repeat(spm_input.shape[0], 1, 1)
    spm = SPM(config_vit, 8, (8, 8))  # input=torch.randn(2, 256, 128, 128) ok
    class_feature, refined_shape_prior = spm(spm_input, learnable_shape_prior)
    print("feature output shape:", class_feature.shape)
    print("shape output shape:", refined_shape_prior.shape)


test_spm_config()
