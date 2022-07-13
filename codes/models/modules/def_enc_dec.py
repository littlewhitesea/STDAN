import torch
import torch.nn as nn
from models.modules.DCNv2_latest.dcn_v2 import DSP_sep2


class STDFA(nn.Module):
    def __init__(self, in_chans, embed_dim, patch_size, topk):
        super(STDFA, self).__init__()

        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.topk = topk

        self.conv_first = nn.Conv2d(
            in_chans, embed_dim, 3, 1, 1
        )  # convert tensor from feature space into embedding space

        self.qkv_maps = QKVmap_generate(embed_dim=embed_dim)

        self.deformable_corr = Deformable_attention(
            embed_dim, kernel_size=patch_size, padding=1, topk=topk
        )

        self.conv_last = nn.Conv2d(
            embed_dim, in_chans, 3, 1, 1
        )  # convert tensor from embedding space into feature space

    def forward(self, fea_full_map):

        B, T, C, H, W = fea_full_map.size()
        fea_full_map = fea_full_map.view(-1, C, H, W)
        res_feature = fea_full_map  # B*T, C, H, W

        fea_full_map_embed = self.conv_first(fea_full_map)
        fea_full_map_embed = fea_full_map_embed.view(B, T, -1, H, W)

        q_maps, key_maps, value_maps = self.qkv_maps(fea_full_map_embed)
        ## B, T, embed_dim, H, W
        ## B, T, embed_dim, H, W
        ## B, T, embed_dim, H, W

        feats_embed = []

        for jj in range(T):

            weight_maps = []
            v_rearrange_maps = []

            for ii_index in range(T):

                if jj != ii_index:
                    weight_ori, v_rearrange_ori = self.deformable_corr(
                        q_maps[:, jj, :, :, :],
                        key_maps[:, ii_index, :, :, :],
                        value_maps[:, ii_index, :, :, :],
                    )
                    # B, H*W
                    # B, embed_dim, H*W

                    weight_maps.append(weight_ori)
                    v_rearrange_maps.append(v_rearrange_ori)

            torch.cuda.empty_cache()
            weight_maps = torch.stack(weight_maps, dim=1)  # B, T-1, H*W
            weight_maps = torch.softmax(weight_maps, dim=1).unsqueeze(
                2
            )  # B, T-1, 1, H*W
            v_rearrange_maps = torch.stack(
                v_rearrange_maps, dim=1
            )  # B, T-1, embed_dim, H*W

            feats_embed.append(
                torch.mul(v_rearrange_maps, weight_maps)
                .sum(dim=1)
                .view(B, self.embed_dim, H, W)
            )  # B, embed_dim, H, W
        del v_rearrange_maps, weight_maps
        feats_embed = torch.stack(feats_embed, dim=1)  # B, T, embed_dim, H, W
        feats_embed = feats_embed.view(-1, self.embed_dim, H, W)  # B*T, embed_dim, H, W

        feats_embed = self.conv_last(feats_embed)  # B*T, C, H, W
        # feats += res_feature  # B*T, C, H, W

        return feats_embed + res_feature


class Deformable_attention(nn.Module):
    def __init__(self, nf, kernel_size=3, padding=1, topk=2, stride=1, bias=None):
        super(Deformable_attention, self).__init__()
        self.topk = topk
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.offset_conv1 = nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True)  # concat for diff
        self.offset_conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.dcn_attention = DSP_sep2(
            nf,
            nf,
            self.kernel_size,
            stride=1,
            padding=1,
            dilation=1,
            deformable_groups=8,
        )

        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, query, key, value):
        offset = torch.cat([query, key], dim=1)  # B, 2*embed_dim, H, W
        offset = self.lrelu(self.offset_conv1(offset))
        offset = self.lrelu(self.offset_conv2(offset))
        B, embed_dim, H, W = offset.size()
        N = self.kernel_size * self.kernel_size

        key_offset = self.dcn_attention(
            key.contiguous(), offset
        )  # B, embed_dim*N, H, W
        key_offset = (
            key_offset.view(B, embed_dim, N, H, W)
            .permute(0, 3, 4, 2, 1)
            .view(B, H * W, N, embed_dim)
        )
        # B, H*W, N, embed_dim

        query = (
            query.view(B, embed_dim, H * W).permute(0, 2, 1).unsqueeze(-1)
        )  # B, H*W, embed_dim, 1

        relevance = torch.matmul(key_offset, query)  # B, H*W, N, 1

        correlation_window, rele_ind = torch.topk(
            relevance, self.topk, dim=2
        )  # B, H*W, topk, 1

        correlation_window = torch.softmax(correlation_window, dim=2)  # B, H*W, topk, 1

        value_offset = self.dcn_attention(
            value.contiguous(), offset
        )  # B, embed_dim*N, H, W
        value_offset = (
            value_offset.view(B, embed_dim, N, H, W)
            .permute(0, 3, 4, 2, 1)
            .view(B, H * W, N, embed_dim)
        )
        # B, H*W, N, embed_dim

        key_offset = key_offset.permute(0, 2, 1, 3).reshape(
            B, N, H * W * embed_dim
        )  # B, N, H*W*embed_dim
        value_offset = value_offset.permute(0, 2, 1, 3).reshape(
            B, N, H * W * embed_dim
        )  # B, N, H*W*embed_dim

        rele_ind = rele_ind.view(B, H * W, self.topk).permute(0, 2, 1)  # B, topk, H*W

        key_rearrange_topk = []
        v_rearrange_topk = []
        for ii_topk in range(self.topk):
            index_temp = rele_ind[:, ii_topk, :].unsqueeze(-1)  # B, H*W, 1
            index = index_temp.repeat(1, 1, embed_dim)  # B, H*W, embed_dim
            index = index.reshape(B, -1).unsqueeze(1)  # B, 1, H*W*embed_dim
            # key_offset = key_offset.permute(0, 3, 2, 1).view(B, N, -1)  # B, N, H*W*embed_dim
            key_rearrange_temp = torch.gather(
                key_offset, 1, index
            )  # B, 1, H*W*embed_dim
            value_rearrange_temp = torch.gather(
                value_offset, 1, index
            )  # B, 1, H*W*embed_dim
            key_rearrange_topk.append(key_rearrange_temp)
            v_rearrange_topk.append(value_rearrange_temp)

        key_rearrange_topk = (
            torch.stack(key_rearrange_topk, dim=1)
            .view(B, self.topk, H * W, embed_dim)
            .permute(0, 2, 3, 1)
        )
        # B, H*W, embed_dim, topk

        v_rearrange_topk = (
            torch.stack(v_rearrange_topk, dim=1)
            .view(B, self.topk, H * W, embed_dim)
            .permute(0, 2, 3, 1)
        )
        # B, H*W, embed_dim, topk

        k_update = torch.matmul(
            key_rearrange_topk, correlation_window
        )  # B, H*W, embed_dim, 1

        v_rearrange_ori = (
            torch.matmul(v_rearrange_topk, correlation_window)
            .view(B, H * W, embed_dim)
            .permute(0, 2, 1)
        )  # B, embed_dim, H*W

        query = query.permute(0, 1, 3, 2)  # B, H*W, 1, embed_dim
        weight_ori = torch.matmul(query, k_update).view(B, H * W)  # B, H*W

        return weight_ori, v_rearrange_ori


class QKVmap_generate(nn.Module):
    def __init__(self, embed_dim):
        super(QKVmap_generate, self).__init__()

        self.embed_dim = embed_dim

        self.w_qs = nn.Linear(embed_dim, embed_dim, bias=True)
        self.w_ks = nn.Linear(embed_dim, embed_dim, bias=True)
        self.w_vs = nn.Linear(embed_dim, embed_dim, bias=True)

    def forward(self, fea_full_map):
        B, T, embed_dim, H, W = fea_full_map.size()

        query_maps = []
        key_maps = []
        value_maps = []

        for jj in range(T):
            fea_query_key_value = fea_full_map[:, jj, :, :, :]
            embed_query_key_value = fea_query_key_value.view(
                B, self.embed_dim, -1
            ).permute(0, 2, 1)
            # B, HxW, embed_dim
            q = (
                self.w_qs(embed_query_key_value)
                .permute(0, 2, 1)
                .view(B, embed_dim, H, W)
            )  # B, embed_dim, H, W
            k = (
                self.w_ks(embed_query_key_value)
                .permute(0, 2, 1)
                .view(B, embed_dim, H, W)
            )  # B, embed_dim, H, W
            v = (
                self.w_vs(embed_query_key_value)
                .permute(0, 2, 1)
                .view(B, embed_dim, H, W)
            )  # B, embed_dim, H, W

            query_maps.append(q)
            key_maps.append(k)
            value_maps.append(v)

        query_maps = torch.stack(query_maps, dim=1)  # B, T, embed_dim, H, W
        key_maps = torch.stack(key_maps, dim=1)  # B, T, embed_dim, H, W
        value_maps = torch.stack(value_maps, dim=1)  # B, T, embed_dim, H, W

        return query_maps, key_maps, value_maps
