""" network architecture for Space-Time Video Super-Resolution """
import math

import torch
import torch.nn as nn

from .dconv import PCD_Align_modified
from .def_enc_dec import STDFA
from .trans_enc_dec import Spatial_decoder



class STTrans2(nn.Module):
    def __init__(
        self,
        scale=1,
        n_inputs=2,
        n_outputs=1,
        nf=64,
        embed_dim=72,
        img_size=32,
        groups=8,
        window_size=8,
    ):
        super(STTrans2, self).__init__()
        self.nf = nf
        self.in_frames = n_inputs
        self.ot_frames = n_outputs

        n_layers = 1
        hidden_dim = []
        for _ in range(n_layers):
            hidden_dim.append(nf)

        self.conv_first = nn.Conv2d(3, nf, 3, 1, 1, bias=True)
        self.fea_L2_conv1 = nn.Conv2d(nf, nf, 3, 2, 1, bias=True)
        self.fea_L2_conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.fea_L3_conv1 = nn.Conv2d(nf, nf, 3, 2, 1, bias=True)
        self.fea_L3_conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.pcd_align_modified = PCD_Align_modified(nf=nf, groups=groups)

        self.feature_extraction = Spatial_decoder(
            in_chans=nf,
            img_size=img_size,
            window_size=8,
            depths=[6, 6],
            embed_dim=embed_dim,
            num_heads=[6, 6],
            mlp_ratio=2,
            resi_connection="1conv",
        )

        self.stdfa = STDFA(in_chans=nf, embed_dim=embed_dim, patch_size=3, topk=2)

        self.feature_recon = Spatial_decoder(
            in_chans=nf,
            img_size=32,
            window_size=8,
            depths=[6, 6, 6, 6, 6, 6],
            embed_dim=embed_dim,
            num_heads=[6, 6, 6, 6, 6, 6],
            mlp_ratio=2,
            resi_connection="1conv",
        )

        #### activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        self.conv_before_upsample = nn.Sequential(
            nn.Conv2d(nf, nf, 3, 1, 1), nn.LeakyReLU(inplace=True)
        )
        self.upsample = Upsample(4, nf)
        self.conv_last = nn.Conv2d(nf, 3, 3, 1, 1)

        self.fusion_f_b = nn.Conv2d(2 * nf, nf, 1, 1, bias=True)

    def forward(self, x):
        B, N, C, H, W = x.size()  # N input video frames
        reverse_idx = list(reversed(range(x.shape[1])))
        ###### original intermediate query frame
        #### extract LR features
        # L1
        L1_fea = self.lrelu(self.conv_first(x.view(-1, C, H, W)))
        L1_fea = self.feature_extraction(L1_fea).contiguous()
        # L2
        L2_fea = self.lrelu(self.fea_L2_conv1(L1_fea))
        L2_fea = self.lrelu(self.fea_L2_conv2(L2_fea))
        # L3
        L3_fea = self.lrelu(self.fea_L3_conv1(L2_fea))
        L3_fea = self.lrelu(self.fea_L3_conv2(L3_fea))

        L1_fea = L1_fea.view(B, N, -1, H, W)
        L2_fea = L2_fea.view(B, N, -1, H // 2, W // 2)
        L3_fea = L3_fea.view(B, N, -1, H // 4, W // 4)

        L1_fea_rev = L1_fea[:, reverse_idx, :, :, :]
        L2_fea_rev = L2_fea[:, reverse_idx, :, :, :]
        L3_fea_rev = L3_fea[:, reverse_idx, :, :, :]

        # forward process
        hid_0 = torch.zeros(B, self.nf, H, W).cuda()
        lr_forward = []

        for idx in range(0, N - 1):
            fea1 = [
                L1_fea[:, idx, :, :, :].clone(),
                L2_fea[:, idx, :, :, :].clone(),
                L3_fea[:, idx, :, :, :].clone(),
            ]
            fea2 = [
                L1_fea[:, idx + 1, :, :, :].clone(),
                L2_fea[:, idx + 1, :, :, :].clone(),
                L3_fea[:, idx + 1, :, :, :].clone(),
            ]
            lr2_fea, hid_2 = self.pcd_align_modified(hid_0, fea1, fea2)
            hid_0 = hid_2
            lr_forward.append(lr2_fea.unsqueeze(1))

        # backward process
        hid_0 = torch.zeros(B, self.nf, H, W).cuda()
        lr_backward = []

        for idx in range(0, N - 1):
            fea1 = [
                L1_fea_rev[:, idx, :, :, :].clone(),
                L2_fea_rev[:, idx, :, :, :].clone(),
                L3_fea_rev[:, idx, :, :, :].clone(),
            ]
            fea2 = [
                L1_fea_rev[:, idx + 1, :, :, :].clone(),
                L2_fea_rev[:, idx + 1, :, :, :].clone(),
                L3_fea_rev[:, idx + 1, :, :, :].clone(),
            ]
            lr2_fea, hid_2 = self.pcd_align_modified(hid_0, fea1, fea2)
            hid_0 = hid_2
            lr_backward.append(lr2_fea.unsqueeze(1))
        del fea1, fea2, hid_0, hid_2, lr2_fea
        lr_forward = torch.cat(lr_forward, dim=1)
        # B, medium_N, nf, H, W
        lr_backward = torch.cat(lr_backward, dim=1)
        # B, medium_N, nf, H, W

        _, medium_N, _, _, _ = lr_backward.size()

        reverse_idx = list(reversed(range(lr_backward.shape[1])))
        lr_backward = lr_backward[:, reverse_idx, :, :, :]

        lr_medium_feat = torch.cat((lr_forward, lr_backward), dim=2)
        lr_medium_feat = self.fusion_f_b(lr_medium_feat.view(B * medium_N, -1, H, W))

        lr_medium_feat = lr_medium_feat.view(B, medium_N, self.nf, H, W)
        to_lstm_fea = []
        del (
            x,
            L2_fea,
            L3_fea,
            L1_fea_rev,
            L2_fea_rev,
            L3_fea_rev,
            lr_forward,
            lr_backward,
        )

        for idx in range(medium_N):
            if idx == 0:
                to_lstm_fea.append(L1_fea[:, 0, :, :, :])
            to_lstm_fea.append(lr_medium_feat[:, idx, :, :, :])
            to_lstm_fea.append(L1_fea[:, idx + 1, :, :, :])

        feats_origin = torch.stack(to_lstm_fea, dim=1)
        B, T, C, H, W = feats_origin.size()

        feats = self.stdfa(feats_origin)
        del (feats_origin, to_lstm_fea, L1_fea, lr_medium_feat)
        torch.cuda.empty_cache()
        # pdb.set_trace()

        feats = self.feature_recon(feats)  # 5661->oom
        feats = self.conv_before_upsample(feats)
        feats = self.conv_last(self.upsample(feats))

        _, _, K, G = feats.size()

        return feats.view(B, T, -1, K, G)


class Upsample(nn.Sequential):
    """Upsample module.

    Args:
        scale (int): Scale factor. Supported scales: 2^n and 3.
        num_feat (int): Channel number of intermediate features.
    """

    def __init__(self, scale, num_feat):
        m = []
        if (scale & (scale - 1)) == 0:  # scale = 2^n
            for _ in range(int(math.log(scale, 2))):
                m.append(nn.Conv2d(num_feat, 4 * num_feat, 3, 1, 1))
                m.append(nn.PixelShuffle(2))
        elif scale == 3:
            m.append(nn.Conv2d(num_feat, 9 * num_feat, 3, 1, 1))
            m.append(nn.PixelShuffle(3))
        else:
            raise ValueError(
                f"scale {scale} is not supported. " "Supported scales: 2^n and 3."
            )
        super(Upsample, self).__init__(*m)
