from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.modules.DCNv2_latest.dcn_v2 import (
      ModulatedDeformConv,
      modulated_deform_conv,
)

from torch.nn.modules.utils import _pair


class ModulatedDeformConvPack(ModulatedDeformConv):
    def __init__(self, *args, **kwargs) -> None:
        super(ModulatedDeformConvPack, self).__init__(*args, **kwargs)

        self.conv_offset = nn.Conv2d(
            self.in_channels,
            self.deformable_groups * 3 * self.kernel_size[0] * self.kernel_size[1],
            kernel_size=self.kernel_size,
            stride=_pair(self.stride),
            padding=_pair(self.padding),
            bias=True,
        )
        self.init_offset()

    def init_offset(self) -> None:
        self.conv_offset.weight.data.zero_()
        self.conv_offset.bias.data.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv_offset(x)
        o1, o2, mask = torch.chunk(out, 3, dim=1)
        offset = torch.cat((o1, o2), dim=1)
        mask = torch.sigmoid(mask)
        return modulated_deform_conv(
            x,
            offset,
            mask,
            self.weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
            self.deformable_groups,
        )


class DCNv2Pack(ModulatedDeformConvPack):
    """Modulated deformable conv for deformable alignment.
    Different from the official DCNv2Pack, which generates offsets and masks
    from the preceding features, this DCNv2Pack takes another different
    features to generate offsets and masks.
    Ref:
        Delving Deep into Deformable Alignment in Video Super-Resolution.
    """

    def forward(self, x: torch.Tensor, feat: torch.Tensor) -> torch.Tensor:
        out = self.conv_offset(feat)
        o1, o2, mask = torch.chunk(out, 3, dim=1)
        offset = torch.cat((o1, o2), dim=1)
        mask = torch.sigmoid(mask)

        offset_absmean = torch.mean(torch.abs(offset))
        if offset_absmean > 100:
            print(f"Offset abs mean is {offset_absmean}, larger than 100.")

        return modulated_deform_conv(
            x,
            offset,
            mask,
            self.weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
            self.deformable_groups,
        )


class PCD_Align(nn.Module):
    """Alignment module using Pyramid, Cascading and Deformable convolution
    with 3 pyramid levels.
    """

    def __init__(self, nf: int = 64, groups: int = 8) -> None:
        super(PCD_Align, self).__init__()

        # fea1
        # L3: level 3, 1/4 spatial size
        self.L3_offset_conv1_1 = nn.Conv2d(
            nf * 2, nf, 3, 1, 1, bias=True
        )  # concat for diff
        self.L3_offset_conv2_1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.L3_dcnpack_1 = DCNv2Pack(
            in_channels=nf,
            out_channels=nf,
            kernel_size=3,
            stride=1,
            padding=1,
            dilation=1,
            deformable_groups=groups,
        )
        # L2: level 2, 1/2 spatial size
        self.L2_offset_conv1_1 = nn.Conv2d(
            nf * 2, nf, 3, 1, 1, bias=True
        )  # concat for diff
        self.L2_offset_conv2_1 = nn.Conv2d(
            nf * 2, nf, 3, 1, 1, bias=True
        )  # concat for offset
        self.L2_offset_conv3_1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.L2_dcnpack_1 = DCNv2Pack(
            in_channels=nf,
            out_channels=nf,
            kernel_size=3,
            stride=1,
            padding=1,
            dilation=1,
            deformable_groups=groups,
        )
        self.L2_fea_conv_1 = nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True)  # concat for fea
        # L1: level 1, original spatial size
        self.L1_offset_conv1_1 = nn.Conv2d(
            nf * 2, nf, 3, 1, 1, bias=True
        )  # concat for diff
        self.L1_offset_conv2_1 = nn.Conv2d(
            nf * 2, nf, 3, 1, 1, bias=True
        )  # concat for offset
        self.L1_offset_conv3_1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.L1_dcnpack_1 = DCNv2Pack(
            in_channels=nf,
            out_channels=nf,
            kernel_size=3,
            stride=1,
            padding=1,
            dilation=1,
            deformable_groups=groups,
        )
        self.L1_fea_conv_1 = nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True)  # concat for fea

        # fea2
        # L3: level 3, 1/4 spatial size
        self.L3_offset_conv1_2 = nn.Conv2d(
            nf * 2, nf, 3, 1, 1, bias=True
        )  # concat for diff
        self.L3_offset_conv2_2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.L3_dcnpack_2 = DCNv2Pack(
            in_channels=nf,
            out_channels=nf,
            kernel_size=3,
            stride=1,
            padding=1,
            dilation=1,
            deformable_groups=groups,
        )
        # L2: level 2, 1/2 spatial size
        self.L2_offset_conv1_2 = nn.Conv2d(
            nf * 2, nf, 3, 1, 1, bias=True
        )  # concat for diff
        self.L2_offset_conv2_2 = nn.Conv2d(
            nf * 2, nf, 3, 1, 1, bias=True
        )  # concat for offset
        self.L2_offset_conv3_2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.L2_dcnpack_2 = DCNv2Pack(
            in_channels=nf,
            out_channels=nf,
            kernel_size=3,
            stride=1,
            padding=1,
            dilation=1,
            deformable_groups=groups,
        )
        self.L2_fea_conv_2 = nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True)  # concat for fea
        # L1: level 1, original spatial size
        self.L1_offset_conv1_2 = nn.Conv2d(
            nf * 2, nf, 3, 1, 1, bias=True
        )  # concat for diff
        self.L1_offset_conv2_2 = nn.Conv2d(
            nf * 2, nf, 3, 1, 1, bias=True
        )  # concat for offset
        self.L1_offset_conv3_2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.L1_dcnpack_2 = DCNv2Pack(
            in_channels=nf,
            out_channels=nf,
            kernel_size=3,
            stride=1,
            padding=1,
            dilation=1,
            deformable_groups=groups,
        )
        self.L1_fea_conv_2 = nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True)  # concat for fea

        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(
        self, fea1: List[torch.Tensor], fea2: List[torch.Tensor]
    ) -> torch.Tensor:
        """align other neighboring frames to the reference frame in the feature level
        fea1, fea2: [L1, L2, L3], each with [B,C,H,W] features
        estimate offset bidirectionally
        """
        y = []
        # param. of fea1
        # L3
        L3_offset = torch.cat([fea1[2], fea2[2]], dim=1)
        L3_offset = self.lrelu(self.L3_offset_conv1_1(L3_offset))
        L3_offset = self.lrelu(self.L3_offset_conv2_1(L3_offset))
        L3_fea = self.lrelu(self.L3_dcnpack_1(fea1[2], L3_offset))
        # L2
        L2_offset = torch.cat([fea1[1], fea2[1]], dim=1)
        L2_offset = self.lrelu(self.L2_offset_conv1_1(L2_offset))
        L3_offset = F.interpolate(
            L3_offset, scale_factor=2, mode="bilinear", align_corners=False
        )
        L2_offset = self.lrelu(
            self.L2_offset_conv2_1(torch.cat([L2_offset, L3_offset * 2], dim=1))
        )
        L2_offset = self.lrelu(self.L2_offset_conv3_1(L2_offset))
        L2_fea = self.L2_dcnpack_1(fea1[1], L2_offset)
        L3_fea = F.interpolate(
            L3_fea, scale_factor=2, mode="bilinear", align_corners=False
        )
        L2_fea = self.lrelu(self.L2_fea_conv_1(torch.cat([L2_fea, L3_fea], dim=1)))
        # L1
        L1_offset = torch.cat([fea1[0], fea2[0]], dim=1)
        L1_offset = self.lrelu(self.L1_offset_conv1_1(L1_offset))
        L2_offset = F.interpolate(
            L2_offset, scale_factor=2, mode="bilinear", align_corners=False
        )
        L1_offset = self.lrelu(
            self.L1_offset_conv2_1(torch.cat([L1_offset, L2_offset * 2], dim=1))
        )
        L1_offset = self.lrelu(self.L1_offset_conv3_1(L1_offset))
        L1_fea = self.L1_dcnpack_1(fea1[0], L1_offset)
        L2_fea = F.interpolate(
            L2_fea, scale_factor=2, mode="bilinear", align_corners=False
        )
        L1_fea = self.L1_fea_conv_1(torch.cat([L1_fea, L2_fea], dim=1))
        y.append(L1_fea)

        # param. of fea2
        # L3
        L3_offset = torch.cat([fea2[2], fea1[2]], dim=1)
        L3_offset = self.lrelu(self.L3_offset_conv1_2(L3_offset))
        L3_offset = self.lrelu(self.L3_offset_conv2_2(L3_offset))
        L3_fea = self.lrelu(self.L3_dcnpack_2(fea2[2], L3_offset))
        # L2
        L2_offset = torch.cat([fea2[1], fea1[1]], dim=1)
        L2_offset = self.lrelu(self.L2_offset_conv1_2(L2_offset))
        L3_offset = F.interpolate(
            L3_offset, scale_factor=2, mode="bilinear", align_corners=False
        )
        L2_offset = self.lrelu(
            self.L2_offset_conv2_2(torch.cat([L2_offset, L3_offset * 2], dim=1))
        )
        L2_offset = self.lrelu(self.L2_offset_conv3_2(L2_offset))
        L2_fea = self.L2_dcnpack_2(fea2[1], L2_offset)
        L3_fea = F.interpolate(
            L3_fea, scale_factor=2, mode="bilinear", align_corners=False
        )
        L2_fea = self.lrelu(self.L2_fea_conv_2(torch.cat([L2_fea, L3_fea], dim=1)))
        # L1
        L1_offset = torch.cat([fea2[0], fea1[0]], dim=1)
        L1_offset = self.lrelu(self.L1_offset_conv1_2(L1_offset))
        L2_offset = F.interpolate(
            L2_offset, scale_factor=2, mode="bilinear", align_corners=False
        )
        L1_offset = self.lrelu(
            self.L1_offset_conv2_2(torch.cat([L1_offset, L2_offset * 2], dim=1))
        )
        L1_offset = self.lrelu(self.L1_offset_conv3_2(L1_offset))
        L1_fea = self.L1_dcnpack_2(fea2[0], L1_offset)
        L2_fea = F.interpolate(
            L2_fea, scale_factor=2, mode="bilinear", align_corners=False
        )
        L1_fea = self.L1_fea_conv_2(torch.cat([L1_fea, L2_fea], dim=1))
        y.append(L1_fea)

        y = torch.cat(y, dim=1)
        return y


class Easy_PCD(nn.Module):
    def __init__(self, nf: int = 64, groups: int = 8) -> None:
        super(Easy_PCD, self).__init__()

        self.fea_L2_conv1 = nn.Conv2d(nf, nf, 3, 2, 1, bias=True)
        self.fea_L2_conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.fea_L3_conv1 = nn.Conv2d(nf, nf, 3, 2, 1, bias=True)
        self.fea_L3_conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.pcd_align = PCD_Align(nf=nf, groups=groups)
        self.fusion = nn.Conv2d(2 * nf, nf, 1, 1, bias=True)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, f1: torch.Tensor, f2: torch.Tensor) -> torch.Tensor:
        # input: extracted features
        # feature shape: f1 == f2 == [B, N, C, H, W]
        L1_fea = torch.stack([f1, f2], dim=1)
        B, N, C, H, W = L1_fea.size()
        L1_fea = L1_fea.view(-1, C, H, W)
        # L2
        L2_fea = self.lrelu(self.fea_L2_conv1(L1_fea))
        L2_fea = self.lrelu(self.fea_L2_conv2(L2_fea))
        # L3
        L3_fea = self.lrelu(self.fea_L3_conv1(L2_fea))
        L3_fea = self.lrelu(self.fea_L3_conv2(L3_fea))

        L1_fea = L1_fea.view(B, N, -1, H, W)
        L2_fea = L2_fea.view(B, N, -1, H // 2, W // 2)
        L3_fea = L3_fea.view(B, N, -1, H // 4, W // 4)

        fea1 = [
            L1_fea[:, 0, :, :, :].clone(),
            L2_fea[:, 0, :, :, :].clone(),
            L3_fea[:, 0, :, :, :].clone(),
        ]
        fea2 = [
            L1_fea[:, 1, :, :, :].clone(),
            L2_fea[:, 1, :, :, :].clone(),
            L3_fea[:, 1, :, :, :].clone(),
        ]
        aligned_fea = self.pcd_align(fea1, fea2)
        fusion_fea = self.fusion(aligned_fea)  # [B, N, C, H, W]
        return fusion_fea


class pcd(nn.Module):
    def __init__(self, nf: int = 64, groups: int = 8):
        super(pcd, self).__init__()
        self.fea_L2_conv1 = nn.Conv2d(nf, nf, 3, 2, 1, bias=True)
        self.fea_L2_conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.fea_L3_conv1 = nn.Conv2d(nf, nf, 3, 2, 1, bias=True)
        self.fea_L3_conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.pcd_align = PCD_Align(nf=nf, groups=groups)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, x: torch.Tensor):
        f1 = x[:, :-1, ...]
        f2 = x[:, 1:, ...]
        L1_fea = torch.cat([f1, f2], dim=1)
        B, N, C, H, W = L1_fea.size()
        # L2
        L2_fea = self.lrelu(self.fea_L2_conv1(L1_fea.view(-1, C, H, W)))
        L2_fea = self.lrelu(self.fea_L2_conv2(L2_fea))
        # L3
        L3_fea = self.lrelu(self.fea_L3_conv1(L2_fea))
        L3_fea = self.lrelu(self.fea_L3_conv2(L3_fea))

        L2_fea = L2_fea.view(B, N, -1, H // 2, W // 2)
        L3_fea = L3_fea.view(B, N, -1, H // 4, W // 4)

        fea1 = [
            L1_fea[:, 0, :, :, :].clone(),
            L2_fea[:, 0, :, :, :].clone(),
            L3_fea[:, 0, :, :, :].clone(),
        ]
        fea2 = [
            L1_fea[:, 1, :, :, :].clone(),
            L2_fea[:, 1, :, :, :].clone(),
            L3_fea[:, 1, :, :, :].clone(),
        ]
        aligned_fea = self.pcd_align(fea1, fea2)
        return aligned_fea


class PCD_Align_modified(nn.Module):
    """Alignment module using Pyramid, Cascading and Deformable convolution
    with 3 pyramid levels.
    """

    def __init__(self, nf=64, groups=8):
        super(PCD_Align_modified, self).__init__()

        # fea1
        # L3: level 3, 1/4 spatial size
        self.L3_offset_conv1_1 = nn.Conv2d(
            nf * 2, nf, 3, 1, 1, bias=True
        )  # concat for diff
        self.L3_offset_conv2_1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.L3_dcnpack_1 = DCNv2Pack(
            in_channels=nf,
            out_channels=nf,
            kernel_size=3,
            stride=1,
            padding=1,
            dilation=1,
            deformable_groups=groups,
        )
        # L2: level 2, 1/2 spatial size
        self.L2_offset_conv1_1 = nn.Conv2d(
            nf * 2, nf, 3, 1, 1, bias=True
        )  # concat for diff
        self.L2_offset_conv2_1 = nn.Conv2d(
            nf * 2, nf, 3, 1, 1, bias=True
        )  # concat for offset
        self.L2_offset_conv3_1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.L2_dcnpack_1 = DCNv2Pack(
            in_channels=nf,
            out_channels=nf,
            kernel_size=3,
            stride=1,
            padding=1,
            dilation=1,
            deformable_groups=groups,
        )
        self.L2_fea_conv_1 = nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True)  # concat for fea
        # L1: level 1, original spatial size
        self.L1_offset_conv1_1 = nn.Conv2d(
            nf * 2, nf, 3, 1, 1, bias=True
        )  # concat for diff
        self.L1_offset_conv2_1 = nn.Conv2d(
            nf * 2, nf, 3, 1, 1, bias=True
        )  # concat for offset
        self.L1_offset_conv3_1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.L1_dcnpack_1 = DCNv2Pack(
            in_channels=nf,
            out_channels=nf,
            kernel_size=3,
            stride=1,
            padding=1,
            dilation=1,
            deformable_groups=groups,
        )
        self.L1_fea_conv_1 = nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True)  # concat for fea

        # fea2
        # L3: level 3, 1/4 spatial size
        self.L3_offset_conv1_2 = nn.Conv2d(
            nf * 2, nf, 3, 1, 1, bias=True
        )  # concat for diff
        self.L3_offset_conv2_2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.L3_dcnpack_2 = DCNv2Pack(
            in_channels=nf,
            out_channels=nf,
            kernel_size=3,
            stride=1,
            padding=1,
            dilation=1,
            deformable_groups=groups,
        )
        # L2: level 2, 1/2 spatial size
        self.L2_offset_conv1_2 = nn.Conv2d(
            nf * 2, nf, 3, 1, 1, bias=True
        )  # concat for diff
        self.L2_offset_conv2_2 = nn.Conv2d(
            nf * 2, nf, 3, 1, 1, bias=True
        )  # concat for offset
        self.L2_offset_conv3_2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.L2_dcnpack_2 = DCNv2Pack(
            in_channels=nf,
            out_channels=nf,
            kernel_size=3,
            stride=1,
            padding=1,
            dilation=1,
            deformable_groups=groups,
        )
        self.L2_fea_conv_2 = nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True)  # concat for fea
        # L1: level 1, original spatial size
        self.L1_offset_conv1_2 = nn.Conv2d(
            nf * 2, nf, 3, 1, 1, bias=True
        )  # concat for diff
        self.L1_offset_conv2_2 = nn.Conv2d(
            nf * 2, nf, 3, 1, 1, bias=True
        )  # concat for offset
        self.L1_offset_conv3_2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.L1_dcnpack_2 = DCNv2Pack(
            in_channels=nf,
            out_channels=nf,
            kernel_size=3,
            stride=1,
            padding=1,
            dilation=1,
            deformable_groups=groups,
        )
        self.L1_fea_conv_2 = nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True)  # concat for fea

        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        self.fusion = nn.Conv2d(2 * nf, nf, 1, 1, bias=True)
        self.conv_h1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv_h2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)

        self.fusion_1x1 = nn.Conv2d(2 * nf, nf, 1, 1, bias=True)

        self.pcd_h = Easy_PCD(nf=nf, groups=groups)

    def forward(self, hid_0, fea1, fea2):
        """align other neighboring frames to the reference frame in the feature level
        fea1, fea2: [L1, L2, L3], each with [B,C,H,W] features
        estimate offset bidirectionally
        hid_0 [B,C,H,W] the hidden state from the previous time step
        """
        y = []

        # y.append(hid_0)
        # param. of fea1
        # L3
        L3_offset = torch.cat([fea1[2], fea2[2]], dim=1)
        L3_offset = self.lrelu(self.L3_offset_conv1_1(L3_offset))
        L3_offset = self.lrelu(self.L3_offset_conv2_1(L3_offset))
        L3_fea = self.lrelu(self.L3_dcnpack_1(fea1[2], L3_offset))
        # L2
        L2_offset = torch.cat([fea1[1], fea2[1]], dim=1)
        L2_offset = self.lrelu(self.L2_offset_conv1_1(L2_offset))
        L3_offset = F.interpolate(
            L3_offset, scale_factor=2, mode="bilinear", align_corners=False
        )
        L2_offset = self.lrelu(
            self.L2_offset_conv2_1(torch.cat([L2_offset, L3_offset * 2], dim=1))
        )
        L2_offset = self.lrelu(self.L2_offset_conv3_1(L2_offset))
        L2_fea = self.L2_dcnpack_1(fea1[1], L2_offset)
        L3_fea = F.interpolate(
            L3_fea, scale_factor=2, mode="bilinear", align_corners=False
        )
        L2_fea = self.lrelu(self.L2_fea_conv_1(torch.cat([L2_fea, L3_fea], dim=1)))
        # L1
        L1_offset = torch.cat([fea1[0], fea2[0]], dim=1)
        L1_offset = self.lrelu(self.L1_offset_conv1_1(L1_offset))
        L2_offset = F.interpolate(
            L2_offset, scale_factor=2, mode="bilinear", align_corners=False
        )
        L1_offset = self.lrelu(
            self.L1_offset_conv2_1(torch.cat([L1_offset, L2_offset * 2], dim=1))
        )
        L1_offset = self.lrelu(self.L1_offset_conv3_1(L1_offset))
        L1_fea = self.L1_dcnpack_1(fea1[0], L1_offset)
        L2_fea = F.interpolate(
            L2_fea, scale_factor=2, mode="bilinear", align_corners=False
        )
        L1_fea = self.L1_fea_conv_1(torch.cat([L1_fea, L2_fea], dim=1))
        y.append(L1_fea)

        # param. of fea2
        # L3
        L3_offset = torch.cat([fea2[2], fea1[2]], dim=1)
        L3_offset = self.lrelu(self.L3_offset_conv1_2(L3_offset))
        L3_offset = self.lrelu(self.L3_offset_conv2_2(L3_offset))
        L3_fea = self.lrelu(self.L3_dcnpack_2(fea2[2], L3_offset))
        # L2
        L2_offset = torch.cat([fea2[1], fea1[1]], dim=1)
        L2_offset = self.lrelu(self.L2_offset_conv1_2(L2_offset))
        L3_offset = F.interpolate(
            L3_offset, scale_factor=2, mode="bilinear", align_corners=False
        )
        L2_offset = self.lrelu(
            self.L2_offset_conv2_2(torch.cat([L2_offset, L3_offset * 2], dim=1))
        )
        L2_offset = self.lrelu(self.L2_offset_conv3_2(L2_offset))
        L2_fea = self.L2_dcnpack_2(fea2[1], L2_offset)
        L3_fea = F.interpolate(
            L3_fea, scale_factor=2, mode="bilinear", align_corners=False
        )
        L2_fea = self.lrelu(self.L2_fea_conv_2(torch.cat([L2_fea, L3_fea], dim=1)))
        # L1
        L1_offset = torch.cat([fea2[0], fea1[0]], dim=1)
        L1_offset = self.lrelu(self.L1_offset_conv1_2(L1_offset))
        L2_offset = F.interpolate(
            L2_offset, scale_factor=2, mode="bilinear", align_corners=False
        )
        L1_offset = self.lrelu(
            self.L1_offset_conv2_2(torch.cat([L1_offset, L2_offset * 2], dim=1))
        )
        L1_offset = self.lrelu(self.L1_offset_conv3_2(L1_offset))
        L1_fea = self.L1_dcnpack_2(fea2[0], L1_offset)
        L2_fea = F.interpolate(
            L2_fea, scale_factor=2, mode="bilinear", align_corners=False
        )
        L1_fea = self.L1_fea_conv_2(torch.cat([L1_fea, L2_fea], dim=1))
        y.append(L1_fea)

        y = torch.cat(y, dim=1)
        y = self.fusion(y)

        hid_0 = self.pcd_h(y, hid_0)

        y = self.fusion_1x1(torch.cat([hid_0, y], dim=1))

        hid_2 = self.conv_h2(self.lrelu(self.conv_h1(y)))

        return y, hid_2

