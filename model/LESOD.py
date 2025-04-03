import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_

from model.EdgeNext.model_13 import edgenext_small
from model.mobilenetv3_13 import mobilenetv3_large

TRAIN_SIZE = 384


class LESOD(nn.Module):
    def __init__(self):
        super().__init__()

        # Backbone
        self.rgb_backbone = edgenext_small()
        self.d_backbone = mobilenetv3_large()

        # Fusion
        self.cim3 = CIM(inc=160, stage=3)
        self.cim2 = CIM(inc=96, stage=2)
        self.cim1 = CIM(inc=48, stage=1)

        # Decoder
        self.mfem2 = MFEM(high_dim=160, low_dim=96, kernal_size=7)
        self.mfem1 = MFEM(high_dim=96, low_dim=48, kernal_size=5)

        # Depth channel transform
        self.d_trans_3 = Trans(40, 160)
        self.d_trans_2 = Trans(24, 96)
        self.d_trans_1 = Trans(16, 48)

        if self.training:
            self.predtrans_3 = nn.Sequential(
                nn.Conv2d(160, 256, 1),
                nn.PixelShuffle(16),
            )
            self.predtrans_2 = nn.Sequential(
                nn.Conv2d(96, 64, 1),
                nn.PixelShuffle(8),
            )
        self.predtrans_1 = nn.Sequential(
            nn.Conv2d(48, 16, 1),
            nn.PixelShuffle(4),
        )

    def forward(self, x_rgb, x_d):
        # rgb
        rgb_1, rgb_2, rgb_3 = self.rgb_backbone(x_rgb)

        # depth
        d_1, d_2, d_3 = self.d_backbone(x_d)

        # depth channel transform
        d_3 = self.d_trans_3(d_3)
        d_2 = self.d_trans_2(d_2)
        d_1 = self.d_trans_1(d_1)

        # Fuse
        fuse_3 = self.cim3(rgb_3, d_3)  # [B, 160, 24, 24]
        fuse_2 = self.cim2(rgb_2, d_2)  # [B, 96, 48, 48]
        fuse_1 = self.cim1(rgb_1, d_1)  # [B, 48, 96, 96]

        # Pred
        xf_2 = self.mfem2(fuse_2, fuse_3)

        xf_1 = self.mfem1(fuse_1, xf_2)
        pred_1 = self.predtrans_1(xf_1)

        if self.training:
            pred_3 = self.predtrans_3(fuse_3)
            pred_2 = self.predtrans_2(xf_2)
            return pred_1, pred_2, pred_3
        else:
            return pred_1


class Trans(nn.Module):
    def __init__(self, inc, outc):
        super().__init__()
        self.trans = nn.Sequential(
            nn.Conv2d(in_channels=inc, out_channels=outc, kernel_size=1, stride=2),
        )
        self.apply(self._init_weights)

    def forward(self, d):
        return self.trans(d)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()


class Fusion_Branch(nn.Module):
    def __init__(self, inc, add_ln=True):
        super().__init__()

        self.add_ln = add_ln

        self.linear = nn.Linear(inc, inc, bias=True)
        # self.silu_1 = nn.SiLU()
        self.silu_1 = nn.Identity()

        self.conv11 = nn.Linear(inc, inc, bias=True)
        # self.silu_2 = nn.SiLU()
        self.silu_2 = nn.Identity()

        self.layernorm = LayerNorm(inc)

        if add_ln:
            self.ln = LayerNorm(inc)

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.flatten(2).permute(0, 2, 1)
        x_clone = x.clone()

        x_clone = self.silu_1(self.linear(x_clone))

        x = self.conv11(x)
        x = self.layernorm(x)
        x = self.silu_2(x)

        x = x * x_clone
        if self.add_ln:
            x = self.ln(x)
        x = x.permute(0, 2, 1).reshape(B, C, H, W)

        return x


class CIM(nn.Module):
    def __init__(self, inc, stage=1):
        super().__init__()
        self.stage = stage
        kernal_size_ls = [3, 5, 7]
        self.conv = DWPWConv(inc * 2, inc, kernel_size=kernal_size_ls[stage - 1], padding=stage)
        self.fusion_branch = Fusion_Branch(inc, add_ln=True)
        self.batchnorm = nn.BatchNorm2d(inc)

        self.dw_1 = nn.Conv2d(inc, inc, kernel_size=3, stride=1, padding=1, groups=inc)
        self.ln_1 = LayerNorm(inc, data_format="channels_first")
        self.pw_1 = nn.Conv2d(inc, inc, 1, 1)

        self.pw = nn.Conv2d(inc * 2, inc, 1, 1)

    def forward(self, rgb, depth):
        # B, C, H, W = rgb.shape

        rgb_depth_multi = rgb * depth
        rgb_depth_multi = self.dw_1(rgb_depth_multi)
        rgb_depth_multi = self.ln_1(rgb_depth_multi)
        rgb_depth_multi = rgb_depth_multi + rgb
        rgb_depth_multi = self.pw_1(rgb_depth_multi)

        rgb_depth_cat = torch.cat((rgb, depth), dim=1)
        rgb_depth_cat = self.conv(rgb_depth_cat)
        rgb_depth_cat = self.fusion_branch(rgb_depth_cat)
        rgb_depth_cat = self.batchnorm(rgb_depth_cat)

        fuse = torch.cat((rgb_depth_multi, rgb_depth_cat), dim=1)
        fuse = self.pw(fuse)
        return fuse


class MFEM(nn.Module):
    def __init__(self, high_dim, low_dim, kernal_size=5):
        super().__init__()
        self.upsample2 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.rc = nn.Sequential(
            nn.Conv2d(in_channels=high_dim, out_channels=high_dim, kernel_size=kernal_size, padding=kernal_size // 2,
                      stride=1,
                      groups=high_dim),
            nn.BatchNorm2d(high_dim),
            nn.GELU(),
            nn.Conv2d(in_channels=high_dim, out_channels=low_dim, kernel_size=1, stride=1),
            nn.BatchNorm2d(low_dim),
            nn.GELU()
        )

        self.rc2 = nn.Sequential(
            nn.Conv2d(in_channels=low_dim * 2, out_channels=low_dim * 2, kernel_size=kernal_size,
                      padding=kernal_size // 2,
                      groups=low_dim * 2),
            nn.BatchNorm2d(low_dim * 2),
            nn.GELU(),
            nn.Conv2d(in_channels=low_dim * 2, out_channels=low_dim, kernel_size=1, stride=1),
            nn.BatchNorm2d(low_dim),
            nn.GELU()
        )
        self.ca = ChannelAttention(low_dim * 2)

        self.ln = LayerNorm(low_dim * 2, data_format="channels_first")

        self.ca_2 = ChannelAttention(low_dim)
        self.conv_11 = nn.Sequential(nn.Conv2d(low_dim * 2, low_dim, kernel_size=1, stride=1, padding=0),
                                     nn.GELU())

        self.apply(self._init_weights)

    def forward(self, x_low, x_high):
        # B, C, H, W = x_low.shape
        x_high = F.interpolate(x_high, size=x_low.shape[2], mode="bilinear", align_corners=True)  # 上采样
        x_high = self.rc(x_high)  # 减少通道数

        high_low_cat = torch.cat((x_high, x_low), dim=1)  # 拼接
        high_low_cat_ca = self.ca(high_low_cat)
        high_low_cat = high_low_cat * high_low_cat_ca
        high_low_cat = self.ln(high_low_cat)
        high_low_cat = self.conv_11(high_low_cat)

        high_low_cat = high_low_cat + x_high
        high_low_cat = high_low_cat * x_low

        high_low_cat_ca_2 = self.ca_2(high_low_cat)
        x_forward = high_low_cat * high_low_cat_ca_2

        return x_forward

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()

        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = max_out
        return self.sigmoid(out)


class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


class DWPWConv(nn.Module):
    def __init__(self, inc, outc, kernel_size=3, padding=1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=inc, out_channels=inc, kernel_size=kernel_size, padding=padding, stride=1,
                      groups=inc),
            nn.BatchNorm2d(inc),
            nn.GELU(),
            nn.Conv2d(in_channels=inc, out_channels=outc, kernel_size=1, stride=1),
            nn.BatchNorm2d(outc),
            nn.GELU()
        )

    def forward(self, x):
        return self.conv(x)


if __name__ == '__main__':
    from utils import param, fps, flops

    ################## LESOD # ##################

    model = LESOD()
    model.load_state_dict(torch.load("../ckps/LESOD/LESOD_best_RGBD.pth"))
    x = torch.randn(1, 3, TRAIN_SIZE, TRAIN_SIZE)
    f_ls = model(x, x)

    for i in f_ls:
        print(i.shape)

    param(model)

    # FPS on CPU
    # fps(model=model, epoch_num=5, size=TRAIN_SIZE, gpu=-1, param_count=2)

    # FPS on GPU
    fps(model=model, epoch_num=5, size=TRAIN_SIZE, gpu=0, param_count=2)

    flops(model, TRAIN_SIZE, gpu=2, param_count=2)
