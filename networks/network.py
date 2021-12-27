import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from common.configs import config
from common.misc import nested_tensor_from_tensor_list
from i3d_backbone import InceptionI3d
from pytorch_model_summary import summary

from networks.feature_pyramid import FPN, MLP, _Unit1D, CoarseNetwork
from networks.position_encoding import PositionEmbeddingLearned
from networks.transformer import Graph_Transformer

num_classes = config['dataset']['num_classes']
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


class Mixup_Branch(nn.Module):
    def __init__(self, in_channels, proposal_channels):
        super(Mixup_Branch, self).__init__()
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

    def forward(self, feature, frame_level_feature):
        '''
        feature: (1, 512, t); t = 126
        frame_level_feature: (1, 512, 256)
        '''
        fm_short = self.cur_point_conv(feature)
        feature = self.lr_conv(feature)

        '''
        inverse_cdf
        '''
        t = feature.size(2)
        max_values = torch.max(frame_level_feature, dim=1)[0]
        sum_value = torch.sum(max_values)
        max_values /= sum_value
        cdf_values = torch.cumsum(max_values, dim=1)[0]  # 256
        cdf_values = (cdf_values * t).int()
        cur_idx = 0
        for i in range(t):
            while len(torch.where(cdf_values == cur_idx)[0]) == 0:
                cur_idx += 1
            idx = torch.where(cdf_values == cur_idx)[0][0]

            if i == 0:
                sampled_feature = frame_level_feature[:, :, idx].unsqueeze(-1)
            else:
                sampled_feature = torch.cat(
                    [sampled_feature, frame_level_feature[:, :, idx].unsqueeze(-1)], dim=-1)
        assert sampled_feature.size(2) == t

        mixed_feature = torch.cat(
            [sampled_feature, feature, fm_short], dim=1)
        mixed_feature = self.proposal_conv(mixed_feature)

        return mixed_feature, feature


class PTN(nn.Module):
    def __init__(self, num_classes, num_queries=126, hidden_dim=256, in_channels=3, training=True):
        super(PTN, self).__init__()
        self.num_classes = num_classes
        self.num_queries = num_queries
        self.backbone = I3D_BackBone(in_channels=in_channels)
        self._training = training
        self.coarseNet = CoarseNetwork(self.num_classes, 1024, 512)
        self.feature_pyramid_net = FPN(self.num_classes, [832, 1024])
        self.poisition_embedding = PositionEmbeddingLearned(
            num_pos_dict=512, num_pos_feats=hidden_dim)

        self.transformer = Graph_Transformer(
            nqueries=num_queries,
            d_model=hidden_dim,
            nhead=4,
            num_encoder_layers=4,
            num_decoder_layers=4,
            dim_feedforward=1024,
            dropout=0,
            activation='leaky_relu',
            normalize_before=True,
            return_intermediate_dec=True)
        self.loc_mixup_branch = Mixup_Branch(512, 512)
        self.conf_mixup_branch = Mixup_Branch(512, 512)

        # self.input_proj = nn.Conv1d(512, hidden_dim, kernel_size=1)
        # self.class_embed = nn.Linear(hidden_dim, num_classes)
        # self.segments_embed = MLP(hidden_dim, hidden_dim, 2, 3)

        self.loc_query_embed = nn.Embedding(num_queries, hidden_dim)
        self.conf_query_embed = nn.Embedding(num_queries, hidden_dim)
        if self._training:
            self.backbone.load_pretrained_weight()

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

    def forward(self, x):
        feat = self.backbone(x)
        loc, conf, priors, start, end, loc_feat, conf_feat, frame_level_feat = self.feature_pyramid_net(
            feat)
        # loc_feat, conf_feat -> B x T x C
        loc_trans_feat, loc_trans_feat_ = self.loc_mixup_branch(
            loc_feat.permute(0, 2, 1), frame_level_feat)
        conf_trans_feat, conf_trans_feat_ = self.conf_mixup_branch(
            conf_feat.permute(0, 2, 1), frame_level_feat)

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

        query_embeds = self.loc_query_embed.weight
        hs, _, edge = self.transformer(src, (mask == 1), query_embeds, pos)

        # hs = hs.squeeze(0)
        # outputs_segments = F.relu(self.segments_embed(hs))
        outputs_segments = self.prop_loc_head(
            hs[-1].permute(0, 2, 1)).permute(0, 2, 1)

        pos = self.poisition_embedding(
            conf_trans_input.tensors, conf_trans_input.mask)
        src, mask = conf_trans_input.tensors, conf_trans_input.mask
        # src = self.input_proj(src)

        query_embeds = self.conf_query_embed.weight
        hs, _, edge = self.transformer(src, (mask == 1), query_embeds, pos)

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
