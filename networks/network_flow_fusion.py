import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from common.configs import config
from common.misc import nested_tensor_from_tensor_list
from i3d_backbone_flow_fusion import InceptionI3d

from networks.feature_pyramid import MLP
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

    def load_pretrained_weight(self, model_path='models/i3d_models/rgb_imagenet.pt'):
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


class MTCA(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(MTCA, self).__init__()
        self.mptc_e1 = MPTC_E(in_channels, out_channels, dilation=2)
        self.mptc_e2 = MPTC_E(out_channels, out_channels, dilation=2 ** 2)
        self.mptc_e3 = MPTC_E(out_channels, out_channels, dilation=2 ** 3)
        self.mptc_e4 = MPTC_E(out_channels, out_channels, dilation=2 ** 4)
        self.mptc_e5 = MPTC_E(out_channels, out_channels, dilation=2 ** 5)
        self.mptc_e6 = MPTC_E(out_channels, out_channels, dilation=2 ** 6)
        self.mptc_e7 = MPTC_E(out_channels, out_channels, dilation=2 ** 7)

        self.mptc_s1 = MPTC_S(in_channels, out_channels)
        self.mptc_s2 = MPTC_S(out_channels, out_channels)
        self.mptc_s3 = MPTC_S(out_channels, out_channels)
        self.mptc_s4 = MPTC_S(out_channels, out_channels)
        self.mptc_s5 = MPTC_S(out_channels, out_channels)
        self.mptc_s6 = MPTC_S(out_channels, out_channels)
        self.mptc_s7 = MPTC_S(out_channels, out_channels)

    def forward(self, x):
        x = self.mptc_s1(self.mptc_e1(x))
        x = self.mptc_s2(self.mptc_e2(x))
        x = self.mptc_s3(self.mptc_e3(x))
        x = self.mptc_s4(self.mptc_e4(x))
        x = self.mptc_s5(self.mptc_e5(x))
        x = self.mptc_s6(self.mptc_e6(x))
        x = self.mptc_s7(self.mptc_e7(x))

        return x


class MPTC_S(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(MPTC_S, self).__init__()
        self.long_range_path = nn.Sequential(
            nn.Conv1d(in_channels, out_channels,
                      kernel_size=3, dilation=3, padding=3),
            nn.BatchNorm1d(out_channels)
        )
        self.short_range_path = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_channels)
        )

        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        long = self.long_range_path(x)
        short = self.short_range_path(x)
        x_self = self.bn(x)

        out = self.relu(long + short + x_self)
        return out


class MPTC_E(nn.Module):
    def __init__(self, in_channels, out_channels, dilation):
        super(MPTC_E, self).__init__()
        self.long_range_path = nn.Sequential(
            nn.Conv1d(in_channels, out_channels,
                      kernel_size=3, dilation=dilation, padding=dilation),
            nn.BatchNorm1d(out_channels)
        )
        self.short_range_path = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_channels)
        )

        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        long = self.long_range_path(x)
        short = self.short_range_path(x)
        x_self = self.bn(x)

        out = self.relu(long + short + x_self)
        return out


class PTN(nn.Module):
    def __init__(self, num_classes, num_queries=126, hidden_dim=256):
        super(PTN, self).__init__()
        self.num_classes = num_classes
        self.num_queries = num_queries
        self.hidden_dim = hidden_dim
        self.rgb_backbone = I3D_BackBone(in_channels=3)
        self.flow_backbone = I3D_BackBone(in_channels=2)

        self.rgb_convs = nn.Sequential(
            nn.Conv1d(1024, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True)
        )

        self.flow_convs = nn.Sequential(
            nn.Conv1d(1024, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True)
        )

        self.mtca = MTCA(512, hidden_dim)

        self.poisition_embedding = PositionEmbeddingLearned(
            num_pos_dict=512, num_pos_feats=hidden_dim)

        self.transformer = Graph_Transformer(
            nqueries=num_queries,
            d_model=hidden_dim,
            nhead=4,
            num_encoder_layers=2,
            num_decoder_layers=1,
            dim_feedforward=1024,
            dropout=0,
            activation='leaky_relu',
            normalize_before=True,
            return_intermediate_dec=True)

        self.input_proj = nn.Conv1d(512, hidden_dim, kernel_size=1)
        self.class_embed = nn.Linear(hidden_dim, num_classes)
        self.segments_embed = MLP(hidden_dim, hidden_dim, 2, 3)
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        self.center_head = nn.Conv1d(
            in_channels=hidden_dim, out_channels=1, kernel_size=3, stride=1, padding=1, bias=True)

        self.rgb_backbone.load_pretrained_weight(
            'models/i3d_models/rgb_imagenet.pt')
        self.flow_backbone.load_pretrained_weight(
            'models/i3d_models/flow_imagenet.pt')

    def forward(self, x, flow):
        num_batch = x.size(0)
        rgb_feat = self.rgb_backbone(x)
        flow_feat = self.flow_backbone(flow)

        rgb_feat = rgb_feat.mean(-1).mean(-1)
        flow_feat = flow_feat.mean(-1).mean(-1)

        rgb_feat = self.rgb_convs(rgb_feat)
        flow_feat = self.flow_convs(flow_feat)

        fusion_feat = torch.cat([rgb_feat, flow_feat], dim=1)
        mtca_out = self.mtca(fusion_feat)

        center = self.center_head(mtca_out).view(
            num_batch, 1, -1).permute(0, 2, 1)
        out_logits = self.class_embed(mtca_out.permute(0, 2, 1))
        out_segments = F.relu(self.segments_embed(mtca_out.permute(0, 2, 1)))

        with torch.no_grad():
            trans_in = nested_tensor_from_tensor_list(mtca_out)
        src, mask = trans_in.tensors, trans_in.mask
        pos = self.poisition_embedding(src, mask)
        src = self.input_proj(src)
        query_embeds = self.query_embed.weight

        hs, _, edge = self.transformer(src, (mask == 1), query_embeds, pos)

        hs = hs.squeeze(0)

        trans_class = self.class_embed(hs)
        trans_segments = F.relu(self.segments_embed(hs))

        out = {
            'coarse_logits': out_logits,
            'coarse_segments': out_segments,
            'refine_logits': trans_class,
            'refine_segments': trans_segments,
            'center': center
        }

        return out
