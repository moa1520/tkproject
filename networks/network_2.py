import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from common.configs import config
from common.misc import nested_tensor_from_tensor_list
from i3d_backbone import InceptionI3d

from networks.feature_pyramid import MLP, _Unit1D
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

    def findNearNum(self, exList, values):
        answer = [0 for _ in range(2)]  # answer 리스트 0으로 초기화

        minValue = min(exList, key=lambda x: abs(x-values))
        minIndex = exList.index(minValue)
        answer[0] = minIndex
        answer[1] = minValue

        return answer

    def forward(self, feature, frame_level_feature):
        '''
        feature: (1, 512, t); t = 100
        frame_level_feature: (1, 512, 256)
        '''
        fm_short = self.cur_point_conv(feature)
        feature = self.lr_conv(feature)

        '''
        inverse_cdf
        '''
        t = feature.size(2)
        # max_values = torch.max(frame_level_feature, dim=1)[0]
        mean_values = torch.mean(frame_level_feature, dim=1)[0]
        sum_value = torch.sum(mean_values)
        mean_values /= sum_value
        cdf_values = torch.cumsum(mean_values, dim=0)
        cdf_values = (cdf_values * t).int()
        cdf_values = torch.clamp(cdf_values, max=t-1).tolist()
        for i in range(t):
            idx, value = self.findNearNum(cdf_values, i)

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

        self.entrance = nn.Sequential(
            nn.Conv3d(1024, 512, kernel_size=[
                      1, 3, 3], stride=[1, 1, 1]),
            nn.GroupNorm(32, 512),
            nn.ReLU(inplace=True)
        )

        self.coarse_regressor = MLP(hidden_dim, hidden_dim, 2, 3)
        self.coarse_classifier = nn.Linear(hidden_dim, num_classes)
        self.refine_regressor = MLP(hidden_dim, hidden_dim, 2, 3)
        self.refine_classifier = nn.Linear(hidden_dim, num_classes)

        self.loc_braches = nn.ModuleList()

        self.loc_braches.append(nn.Sequential(
            nn.Conv1d(512, 512, kernel_size=1, stride=1),
            nn.GroupNorm(32, 512),
            nn.ReLU(inplace=True)
        ))

        for i in range(3):
            self.loc_braches.append(nn.Sequential(
                nn.Conv1d(512, 512, kernel_size=3,
                          stride=1, dilation=2 ** (i + 1)),
                nn.GroupNorm(32, 512),
                nn.ReLU(inplace=True)
            ))

        self.conf_branches = nn.ModuleList()

        self.conf_branches.append(nn.Sequential(
            nn.Conv1d(512, 512, kernel_size=1, stride=1),
            nn.GroupNorm(32, 512),
            nn.ReLU(inplace=True)
        ))

        for i in range(3):
            self.conf_branches.append(nn.Sequential(
                nn.Conv1d(512, 512, kernel_size=3,
                          stride=1, dilation=2 ** (i + 1)),
                nn.GroupNorm(32, 512),
                nn.ReLU(inplace=True)
            ))

        self.deconv = nn.Sequential(
            _Unit1D(512, 512, 3, activation_fn=None),
            nn.GroupNorm(32, 512),
            nn.ReLU(inplace=True),
            _Unit1D(512, 512, 3, activation_fn=None),
            nn.GroupNorm(32, 512),
            nn.ReLU(inplace=True),
            _Unit1D(512, 512, 1, activation_fn=None),
            nn.GroupNorm(32, 512),
            nn.ReLU(inplace=True),
        )

        self.priors = []
        t = [32, 28, 24, 16]
        for i in range(4):
            self.priors.append(
                torch.Tensor([[(c + 0.5) / t[i]]
                             for c in range(t[i])]).view(-1, 1)
            )

    def forward(self, x):
        loc_feats = []
        conf_feats = []
        loc_memories = []
        conf_memories = []
        coarse_boundaries = []
        coarse_classes = []
        frame_num = x.size(2)
        # x: 1 x 3 x 256 x 96 x 96
        feat = self.backbone(x)
        # feat['Mixed_5c']: 1 x 1024 x 32 x 3 x 3
        feat_ = self.entrance(
            feat['Mixed_5c']).squeeze(-1).squeeze(-1)  # 1 x 512 x 32

        frame_level_feat = feat_.unsqueeze(-1)
        frame_level_feat = F.interpolate(
            frame_level_feat, [frame_num, 1]).squeeze(-1)
        frame_level_feat = self.deconv(frame_level_feat)
        start_feat = frame_level_feat[:, :256]
        end_feat = frame_level_feat[:, 256:]
        start = start_feat.permute(0, 2, 1).contiguous()
        end = end_feat.permute(0, 2, 1).contiguous()

        for i in range(4):
            loc_feats.append(self.loc_braches[i](feat_))  # 32, 28, 24, 16
            conf_feats.append(self.conf_branches[i](feat_))

        for i in range(4):
            with torch.no_grad():
                loc_trans_input = nested_tensor_from_tensor_list(
                    loc_feats[i])  # (n, t, c) -> (n, c, t)
                conf_trans_input = nested_tensor_from_tensor_list(
                    conf_feats[i])

            pos = self.poisition_embedding(
                loc_trans_input.tensors, loc_trans_input.mask)
            src, mask = loc_trans_input.tensors, loc_trans_input.mask

            query_embeds = self.loc_query_embed.weight
            hs, memory, edge = self.transformer(
                src, (mask == 1), query_embeds, pos)
            loc_memories.append(hs[-1])

            coarse_boundary = self.coarse_regressor(hs[-1])
            coarse_boundaries.append(coarse_boundary)

            pos = self.poisition_embedding(
                conf_trans_input.tensors, conf_trans_input.mask)
            src, mask = conf_trans_input.tensors, conf_trans_input.mask

            query_embeds = self.conf_query_embed.weight
            hs, memory, edge = self.transformer(
                src, (mask == 1), query_embeds, pos)
            conf_memories.append(hs[-1])

            coarse_class = self.coarse_classifier(hs[-1])
            coarse_classes.append(coarse_class)

        loc_feat = torch.cat(loc_memories, dim=1).permute(0, 2, 1)
        conf_feat = torch.cat(conf_memories, dim=1).permute(0, 2, 1)
        loc_mixup_feat, loc_mixup_feat_ = self.loc_mixup_branch(
            loc_feat, frame_level_feat)  # N x C x T
        conf_mixup_feat, conf_mixup_feat_ = self.conf_mixup_branch(
            conf_feat, frame_level_feat)  # N x C x T

        refined_boundary = self.refine_regressor(
            loc_mixup_feat.permute(0, 2, 1))
        refined_class = self.refine_classifier(
            conf_mixup_feat.permute(0, 2, 1))

        center = self.center_head(loc_mixup_feat).permute(0, 2, 1)
        coarse_boundaries = torch.cat(coarse_boundaries, dim=1)
        coarse_classes = torch.cat(coarse_classes, dim=1)
        priors = torch.cat(self.priors, 0).to(
            coarse_boundaries.device).unsqueeze(0)

        ndim = loc_mixup_feat_.size(1) // 2
        start_loc_prop = loc_mixup_feat_[
            :, :ndim, ].permute(0, 2, 1).contiguous()
        end_loc_prop = loc_mixup_feat_[
            :, ndim:, ].permute(0, 2, 1).contiguous()
        start_conf_prop = conf_mixup_feat_[
            :, :ndim, ].permute(0, 2, 1).contiguous()
        end_conf_prop = conf_mixup_feat_[
            :, ndim:, ].permute(0, 2, 1).contiguous()

        return {
            'loc': coarse_boundaries,
            'conf': coarse_classes,
            'refined_loc': refined_boundary,
            'refined_cls': refined_class,
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
