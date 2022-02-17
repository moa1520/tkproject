import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from common.configs import config
from common.misc import nested_tensor_from_tensor_list
from i3d_backbone import InceptionI3d

from networks.boundary_pooling import BoundaryMaxPooling
from networks.feature_pyramid_anet import FPN, CoarseNetwork, _Unit1D
from networks.position_encoding import PositionEmbeddingLearned
from networks.transformer_unet_anet import Transformer

num_classes = 2
freeze_bn = config['model']['freeze_bn']
freeze_bn_affine = config['model']['freeze_bn_affine']


class I3D_BackBone(nn.Module):
    def __init__(self, final_endpoint='Mixed_5c', name='inception_i3d', in_channels=3,
                 freeze_bn=freeze_bn, freeze_bn_affine=freeze_bn_affine):
        super(I3D_BackBone, self).__init__()
        self._model = InceptionI3d(final_endpoint=final_endpoint,
                                   name=name,
                                   in_channels=in_channels)
        self._model.build()
        self._freeze_bn = freeze_bn
        self._freeze_bn_affine = freeze_bn_affine

    def load_pretrained_weight(self, model_path=config['model']['backbone_model']):
        self._model.load_state_dict(torch.load(model_path), strict=False)

    def train(self, mode=True):
        super(I3D_BackBone, self).train(mode)
        if self._freeze_bn and mode:
            # print('freeze all BatchNorm3d in I3D backbone.')
            for name, m in self._model.named_modules():
                if isinstance(m, nn.BatchNorm3d):
                    # print('freeze {}.'.format(name))
                    m.eval()
                    if self._freeze_bn_affine:
                        m.weight.requires_grad_(False)
                        m.bias.requires_grad_(False)

    def forward(self, x):
        return self._model.extract_features(x)


class Inverse_CDF_Sampling(nn.Module):
    def __init__(self):
        super(Inverse_CDF_Sampling, self).__init__()

    def findNearNum(self, exList, values):
        answer = [0 for _ in range(2)]  # answer 리스트 0으로 초기화

        minValue = min(exList, key=lambda x: abs(x-values))
        minIndex = exList.index(minValue)
        answer[0] = minIndex
        answer[1] = minValue

        return answer

    def forward(self, frame_level_feature, t_size=126):
        mean_values = torch.mean(frame_level_feature, dim=1)[0]
        sum_value = torch.sum(mean_values)
        mean_values /= sum_value
        cdf_values = torch.cumsum(mean_values, dim=0)
        cdf_values = (cdf_values * t_size).int()
        cdf_values = torch.clamp(cdf_values, max=t_size-1).tolist()
        idx_list = []
        for i in range(t_size):
            idx, value = self.findNearNum(cdf_values, i)
            idx_list.append(idx)

            if i == 0:
                sampled_feature = frame_level_feature[:, :, idx].unsqueeze(-1)
            else:
                sampled_feature = torch.cat(
                    [sampled_feature, frame_level_feature[:, :, idx].unsqueeze(-1)], dim=-1)
        assert sampled_feature.size(2) == t_size

        return sampled_feature


class Mixup_Branch(nn.Module):
    def __init__(self, in_channels, proposal_channels):
        super(Mixup_Branch, self).__init__()
        self.boundary_max_pooling = BoundaryMaxPooling()
        self.inverse_cdf_sampling = Inverse_CDF_Sampling()
        self.cur_point_conv = nn.Sequential(
            _Unit1D(in_channels=in_channels,
                    output_channels=proposal_channels,
                    kernel_shape=1,
                    activation_fn=None),
            nn.GroupNorm(32, proposal_channels),
            nn.ReLU(inplace=True)
        )
        self.lr_conv = nn.Sequential(
            _Unit1D(in_channels=in_channels,
                    output_channels=proposal_channels * 2,
                    kernel_shape=1,
                    activation_fn=None),
            nn.GroupNorm(32, proposal_channels * 2),
            nn.ReLU(inplace=True)
        )
        self.proposal_conv = nn.Sequential(
            _Unit1D(in_channels=proposal_channels * 4,
                    output_channels=in_channels,
                    kernel_shape=1,
                    activation_fn=None),
            nn.GroupNorm(32, in_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, feature, frame_level_feature, segments):
        '''
        feature: (1, 512, t); t = 126
        frame_level_feature: (1, 512, 256)
        segments: (1, 126, 4)
        '''
        fm_short = self.cur_point_conv(feature)  # 1 x 512 x 126
        feature = self.lr_conv(feature)  # 1 x 1024 x 126
        prop_feature = self.boundary_max_pooling(
            feature, segments, max_len=feature.size(2))  # 1 x 1024 x 126
        '''
        inverse_cdf
        '''
        t = feature.size(2)
        sampled_feature = self.inverse_cdf_sampling(
            frame_level_feature, t_size=t)  # 1 x 512 x 126

        mixed_feature = torch.cat(
            [sampled_feature, prop_feature, fm_short], dim=1)
        mixed_feature = self.proposal_conv(mixed_feature)

        return mixed_feature, feature


class MMS(nn.Module):
    def __init__(self, in_channels, proposal_channels):
        super(MMS, self).__init__()
        self.inverse_cdf_sampling = Inverse_CDF_Sampling()
        self.lr_conv = nn.Sequential(
            _Unit1D(in_channels=in_channels,
                    output_channels=proposal_channels * 2,
                    kernel_shape=1,
                    activation_fn=None),
            nn.GroupNorm(32, proposal_channels * 2),
            nn.ReLU(inplace=True)
        )
        self.proposal_conv = nn.Sequential(
            _Unit1D(in_channels=proposal_channels * 4,
                    output_channels=in_channels,
                    kernel_shape=1,
                    activation_fn=None),
            nn.GroupNorm(32, in_channels),
            nn.ReLU(inplace=True)
        )
        self.sample_conv = nn.Sequential(
            _Unit1D(in_channels=proposal_channels,
                    output_channels=proposal_channels,
                    kernel_shape=1,
                    activation_fn=None),
            nn.GroupNorm(32, proposal_channels),
            nn.ReLU(inplace=True)
        )
        self.cur_conv = nn.Sequential(
            _Unit1D(
                in_channels=proposal_channels,
                output_channels=proposal_channels,
                kernel_shape=1,
                activation_fn=None
            ),
            nn.GroupNorm(32, proposal_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, feature, frame_level_feature):
        '''
        feature: (1, 512, t); t = 126
        frame_level_feature: (1, 512, 256)
        segments: (1, 126, 4)
        '''
        cur_feature = self.cur_conv(feature)  # 1 x 512 x 126
        lr_feature = self.lr_conv(feature)  # 1 x 1024 x 126

        # t = [64, 32, 16, 8, 4, 2]
        t = [96, 48, 24, 12, 6, 3]
        # Conv(1 x 512 x [64, 32, 16, 8, 4, 2]) -> 1 x 512 x [64, 32, 16, 8, 4, 2]
        sampled_features = [self.sample_conv(
            self.inverse_cdf_sampling(frame_level_feature, t_size=x)) for x in t]
        sampled_feature = torch.cat(sampled_features, dim=2)
        mixed_feature = torch.cat(
            [sampled_feature, lr_feature, cur_feature], dim=1)  # 1 x 204   8 x 126
        mixed_feature = self.proposal_conv(mixed_feature)

        return mixed_feature, lr_feature


class PTN(nn.Module):
    def __init__(self, num_classes, num_queries=126, hidden_dim=256, in_channels=3, training=True):
        super(PTN, self).__init__()
        self.num_classes = num_classes
        self.num_queries = num_queries
        self.backbone = I3D_BackBone(in_channels=in_channels)
        self._training = training
        self.coarseNet = CoarseNetwork(self.num_classes, 1024, 512)
        self.feature_pyramid_net = FPN(self.num_classes, [832, 1024])
        self.reset_params()
        self.poisition_embedding = PositionEmbeddingLearned(
            num_pos_dict=512, num_pos_feats=hidden_dim)
        self.num_heads = config['training']['num_heads']
        self.inverse_cdf_sampling = Inverse_CDF_Sampling()

        self.transformer = Transformer(
            nqueries=num_queries,
            d_model=hidden_dim,
            nhead=self.num_heads,
            num_encoder_layers=1,
            num_decoder_layers=1,
            dim_feedforward=1024,
            dropout=0,
            activation='relu',
            normalize_before=True,
            return_intermediate_dec=True)
        self.loc_mixup_branch = Mixup_Branch(512, 512)
        self.conf_mixup_branch = Mixup_Branch(512, 512)

        # self.input_proj = nn.Conv1d(512, hidden_dim, kernel_size=1)
        # self.class_embed = nn.Linear(hidden_dim, num_classes)
        # self.segments_embed = MLP(hidden_dim, hidden_dim, 2, 3)

        # self.loc_query_embed = nn.Embedding(num_queries, hidden_dim)
        # self.conf_query_embed = nn.Embedding(num_queries, hidden_dim)
        if self._training:
            self.backbone.load_pretrained_weight()

        self.boundary_max_pooling = BoundaryMaxPooling()

        self.center_head = _Unit1D(
            in_channels=512,
            output_channels=1,
            kernel_shape=3,
            stride=1,
            use_bias=True,
            activation_fn=None
        )

        self.prop_loc_head = _Unit1D(
            in_channels=hidden_dim,
            output_channels=2,
            kernel_shape=1,
            activation_fn=None)

        self.prop_conf_head = _Unit1D(
            in_channels=hidden_dim,
            output_channels=num_classes,
            kernel_shape=1,
            activation_fn=None
        )
        self.scales = [1, 768/189, 768/189]

    @staticmethod
    def weight_init(m):
        def glorot_uniform_(tensor):
            fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(tensor)
            scale = 1.0
            scale /= max(1., (fan_in + fan_out) / 2.)
            limit = np.sqrt(3.0 * scale)
            return nn.init._no_grad_uniform_(tensor, -limit, limit)

        if isinstance(m, nn.Conv1d) or isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv3d) \
                or isinstance(m, nn.ConvTranspose3d):
            glorot_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def reset_params(self):
        for i, m in enumerate(self.modules()):
            self.weight_init(m)

    def forward(self, x, proposals=None, ssl=False):
        feat = self.backbone(x)
        if ssl:
            trip = []
            loc, conf, priors, loc_feat, conf_feat, frame_level_feat = self.feature_pyramid_net(
                feat, ssl)

            t = loc.size(1)
            with torch.no_grad():
                segments = loc / \
                    config['dataset']['training']['clip_length'] * t
                new_priors = torch.round(priors[:, :, :1] * t - 0.5)
                plen = segments[:, :, :1] + segments[:, :, 1:]
                plen = torch.clamp(plen / 5.0, min=1.0)
                l_segment = new_priors - segments[:, :, :1]
                r_segment = new_priors + segments[:, :, 1:]
                segments = torch.cat([
                    torch.round(l_segment - plen),
                    torch.round(l_segment + plen),
                    torch.round(r_segment - plen),
                    torch.round(r_segment + plen)
                ], dim=-1)

            _, loc_trans_feat_ = self.loc_mixup_branch(
                loc_feat.permute(0, 2, 1), frame_level_feat, segments)
            _, conf_trans_feat_ = self.conf_mixup_branch(
                conf_feat.permute(0, 2, 1), frame_level_feat, segments)
            trip.append(frame_level_feat.clone())
            trip.extend([loc_trans_feat_.clone(), conf_trans_feat_.clone()])

            decoded_segments = proposals[0].unsqueeze(0)
            plen = decoded_segments[:, :, 1:] - \
                decoded_segments[:, :, :1] + 1.0
            plen = torch.clamp(plen / 5.0, min=1.0)
            frame_segments = torch.cat([
                torch.round(decoded_segments[:, :, :1] - plen),
                torch.round(decoded_segments[:, :, :1] + plen),
                torch.round(decoded_segments[:, :, 1:] - plen),
                torch.round(decoded_segments[:, :, 1:] + plen)
            ], dim=-1)
            anchor, positive, negative = [], [], []
            max_lens = [768, 189, 189]
            for i in range(3):
                bound_feat = self.boundary_max_pooling(
                    trip[i], frame_segments / self.scales[i], max_lens[i])
                # for triplet loss
                ndim = bound_feat.size(1) // 2
                anchor.append(bound_feat[:, ndim:, 0])
                positive.append(bound_feat[:, :ndim, 1])
                negative.append(bound_feat[:, :ndim, 2])

            return anchor, positive, negative

        else:
            loc, conf, priors, start, end, loc_feat, conf_feat, frame_level_feat = self.feature_pyramid_net(
                feat)

            t = loc.size(1)
            with torch.no_grad():
                segments = loc / \
                    config['dataset']['training']['clip_length'] * t
                new_priors = torch.round(priors[:, :, :1] * t - 0.5)
                plen = segments[:, :, :1] + segments[:, :, 1:]
                plen = torch.clamp(plen / 5.0, min=1.0)
                l_segment = new_priors - segments[:, :, :1]
                r_segment = new_priors + segments[:, :, 1:]
                segments = torch.cat([
                    torch.round(l_segment - plen),
                    torch.round(l_segment + plen),
                    torch.round(r_segment - plen),
                    torch.round(r_segment + plen)
                ], dim=-1)

            # loc_feat, conf_feat -> B x T x C
            loc_trans_feat, loc_trans_feat_ = self.loc_mixup_branch(
                loc_feat.permute(0, 2, 1), frame_level_feat, segments)
            conf_trans_feat, conf_trans_feat_ = self.conf_mixup_branch(
                conf_feat.permute(0, 2, 1), frame_level_feat, segments)

            center = self.center_head(loc_trans_feat).permute(0, 2, 1)

            ndim = loc_trans_feat_.size(1) // 2
            start_loc_prop = loc_trans_feat_[
                :, :ndim, ].permute(0, 2, 1).contiguous()
            end_loc_prop = loc_trans_feat_[
                :, ndim:, ].permute(0, 2, 1).contiguous()
            start_conf_prop = conf_trans_feat_[
                :, :ndim, ].permute(0, 2, 1).contiguous()
            end_conf_prop = conf_trans_feat_[
                :, ndim:, ].permute(0, 2, 1).contiguous()

            with torch.no_grad():
                loc_trans_input = nested_tensor_from_tensor_list(
                    loc_trans_feat)  # (n, t, c) -> (n, c, t)
                conf_trans_input = nested_tensor_from_tensor_list(
                    conf_trans_feat)

            pos = self.poisition_embedding(
                loc_trans_input.tensors, loc_trans_input.mask)
            src, mask = loc_trans_input.tensors, loc_trans_input.mask
            # src = self.input_proj(src)

            # query_embeds = self.loc_query_embed.weight.unsqueeze(1)
            query_embeds = loc_trans_feat.permute(2, 0, 1)
            hs = self.transformer(src, (mask == 1), query_embeds, pos)

            # hs = hs.squeeze(0)
            # outputs_segments = F.relu(self.segments_embed(hs))
            outputs_segments = self.prop_loc_head(
                hs[-1].permute(0, 2, 1)).permute(0, 2, 1)

            pos = self.poisition_embedding(
                conf_trans_input.tensors, conf_trans_input.mask)
            src, mask = conf_trans_input.tensors, conf_trans_input.mask
            # src = self.input_proj(src)

            # query_embeds = self.conf_query_embed.weight.unsqueeze(1)
            query_embeds = conf_trans_feat.permute(2, 0, 1)
            hs = self.transformer(src, (mask == 1), query_embeds, pos)

            # hs = hs.squeeze(0)
            # outputs_class = self.class_embed(hs)
            outputs_class = self.prop_conf_head(
                hs[-1].permute(0, 2, 1)).permute(0, 2, 1)

            return {
                'loc': loc,
                'conf': conf,
                'refined_loc': outputs_segments,
                'refined_cls': outputs_class,
                'center': center,
                'priors': priors,
                'start': start,
                'end': end,
                'frame_level_feats': frame_level_feat,
                'start_loc_prop': start_loc_prop,
                'end_loc_prop': end_loc_prop,
                'start_conf_prop': start_conf_prop,
                'end_conf_prop': end_conf_prop
            }


if __name__ == '__main__':
    net = PTN(in_channels=3, training=True)
    input = np.load(
        'datasets/thumos14/test_npy/video_test_0000006.npy')

    input = torch.from_numpy(input).float().permute(
        3, 0, 1, 2).unsqueeze(0)  # 1 x 3 x 670 x 112 x 112
    print(input.shape)

    output = net(input)
    print(output.shape)  # 1 x 512 x 83
