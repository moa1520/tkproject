import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from common.configs import config
from common.misc import (collate_fn, inverse_sigmoid,
                         nested_tensor_from_tensor_list)
from i3d_backbone import InceptionI3d
from pytorch_model_summary import summary

from networks.config import cfg
from networks.feature_pyramid import FPN, MLP, CoarseNetwork
from networks.position_encoding import PositionEmbeddingLearned
from networks.transformer import Graph_Transformer
# from networks.ops.roi_align import ROIAlign
# from networks.transformer import Transformer


class I3D_BackBone(nn.Module):
    def __init__(self, in_channels=3, end_point='Mixed_5c'):
        super(I3D_BackBone, self).__init__()
        self._model = InceptionI3d(
            final_endpoint=end_point, name='inception_i3d', in_channels=in_channels)
        self._model.build()

    def load_pretrained_weight(self, model_path='models/i3d_models/rgb_imagenet.pt'):
        self._model.load_state_dict(torch.load(model_path), strict=False)

    def forward(self, x):
        return self._model.extract_features(x)


class PTN(nn.Module):
    def __init__(self, num_classes, num_queries=500, hidden_dim=256, in_channels=3, training=True):
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
            nqueries=config['training']['num_queries'],
            d_model=config['training']['hidden_dim'],
            nhead=8,
            num_encoder_layers=2,
            num_decoder_layers=4,
            dim_feedforward=2048,
            dropout=0.1,
            activation='relu',
            normalize_before=True,
            return_intermediate_dec=True)

        # self.transformer = Transformer(
        #     d_model=hidden_dim,
        #     nhead=8, num_encoder_layers=2,
        #     num_decoder_layers=4, dim_feedforward=2048,
        #     dropout=0.1, activation='relu',
        #     return_intermediate_dec=True, num_feature_levels=1,
        #     dec_n_points=4, enc_n_points=4)

        self.input_proj = nn.Conv1d(512, hidden_dim, kernel_size=1)
        self.class_embed = nn.Linear(hidden_dim, num_classes + 1)
        self.segments_embed = MLP(hidden_dim, hidden_dim, 2, 3)

        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        # for proj in self.input_proj:
        #     nn.init.xavier_uniform_(proj[0].weight, gain=1)
        #     nn.init.constant_(proj[0].bias, 0)
        # prior_prob = 0.01
        # bias_value = -math.log((1 - prior_prob) / prior_prob)
        # self.class_embed.bias.data = torch.ones(num_classes) * bias_value
        # nn.init.constant_(self.segment_embed.layers[-1].weight.data, 0)
        # nn.init.constant_(self.segment_embed.layers[-1].bias.data, 0)
        # if self._training:
        #     self.backbone.load_pretrained_weight()

        # num_pred = self.transformer.decoder.num_layers  # number of decoder layer
        # nn.init.constant_(
        #     self.segment_embed.layers[-1].bias.data[1:], -2.0)
        # self.class_embed = nn.ModuleList(
        #     [self.class_embed for _ in range(num_pred)])
        # self.segment_embed = nn.ModuleList(
        #     [self.segment_embed for _ in range(num_pred)])
        # self.transformer.decoder.segment_embed = None

    #     if cfg.ACTIONNESS_REG:
    #         # RoIAlign params
    #         self.roi_size = 16
    #         self.roi_scale = 0
    #         self.roi_extractor = ROIAlign(self.roi_size, self.roi_scale)
    #         self.actionness_pred = nn.Sequential(
    #             nn.Linear(self.roi_size * hidden_dim, hidden_dim),
    #             nn.ReLU(inplace=True),
    #             nn.Linear(hidden_dim, hidden_dim),
    #             nn.ReLU(inplace=True),
    #             nn.Linear(hidden_dim, 1),
    #             nn.Sigmoid()
    #         )

    # def _to_roi_align_format(self, rois, T, scale_factor=1):
    #     '''Convert RoIs to RoIAlign format.
    #     Params:
    #         RoIs: normalized segments coordinates, shape (batch_size, num_segments, 4)
    #         T: length of the video feature sequence
    #     '''
    #     # transform to absolute axis
    #     B, N = rois.shape[:2]
    #     rois_center = rois[:, :, 0:1]
    #     rois_size = rois[:, :, 1:2] * scale_factor
    #     rois_abs = torch.cat(
    #         (rois_center - rois_size/2, rois_center + rois_size/2), dim=2) * T
    #     # expand the RoIs
    #     rois_abs = torch.clamp(rois_abs, min=0, max=T)  # (N, T, 2)
    #     # add batch index
    #     batch_ind = torch.arange(0, B).view((B, 1, 1)).to(rois_abs.device)
    #     batch_ind = batch_ind.repeat(1, N, 1)
    #     rois_abs = torch.cat((batch_ind, rois_abs), dim=2)
    #     # NOTE: stop gradient here to stablize training
    #     return rois_abs.view((B*N, 3)).detach()

    def forward(self, x):
        feat = self.backbone(x)
        loc, conf, center, priors, start, end, loc_feats, conf_feats = self.feature_pyramid_net(
            feat)

        with torch.no_grad():
            loc_feats_ = nested_tensor_from_tensor_list(
                loc_feats.permute(0, 2, 1))  # (n, t, c) -> (n, c, t)

        pos = self.poisition_embedding(loc_feats_.tensors, loc_feats_.mask)
        src, mask = loc_feats_.tensors, loc_feats_.mask
        src = self.input_proj(src)

        query_embeds = self.query_embed.weight
        hs, _, edge = self.transformer(src, (mask == 1), query_embeds, pos)

        outputs_class = self.class_embed(hs)
        outputs_segments = F.relu(self.segments_embed(hs))

        out = {'pred_logits': outputs_class[-1],
               'pred_segments': outputs_segments[-1], 'edges': edge}

        # hs, init_reference, inter_reference, memory = self.transformer(
        #     srcs, masks, pos, query_embeds)

        # outputs_classes = []
        # outputs_coords = []
        # # gather outputs from each decoder layers
        # for lvl in range(hs.shape[0]):
        #     if lvl == 0:
        #         reference = init_reference
        #     else:
        #         reference = inter_reference[lvl - 1]

        #     reference = inverse_sigmoid(reference)
        #     outputs_class = self.class_embed[lvl](hs[lvl])
        #     tmp = self.segment_embed[lvl](hs[lvl])
        #     # the l-th layer (l >= 2)
        #     if reference.shape[-1] == 2:
        #         tmp += reference
        #     # the last layer
        #     else:
        #         assert reference.shape[-1] == 1
        #         tmp[..., 0] += reference[..., 0]
        #     outputs_coord = tmp.sigmoid()
        #     outputs_classes.append(outputs_class)
        #     outputs_coords.append(outputs_coord)
        # outputs_class = torch.stack(outputs_classes)
        # outputs_coord = torch.stack(outputs_coords)

        # if not cfg.ACTIONNESS_REG:
        #     out = {'pred_logits': outputs_class[-1],
        #            'pred_segments': outputs_coord[-1]}
        # else:
        #     # perform RoIAlign
        #     B, N = outputs_coord[-1].shape[:2]
        #     origin_feat = memory

        #     rois = self._to_roi_align_format(
        #         outputs_coord[-1], origin_feat.shape[2], scale_factor=1.5)
        #     roi_features = self.roi_extractor(origin_feat, rois)
        #     roi_features = roi_features.view((B, N, -1))
        #     pred_actionness = self.actionness_pred(roi_features)

        #     last_layer_cls = outputs_class[-1]
        #     last_layer_reg = outputs_coord[-1]

        #     out = {'pred_logits': last_layer_cls,
        #            'pred_segments': last_layer_reg, 'pred_actionness': pred_actionness}

        return {
            'loc': loc,
            'conf': conf,
            'center': center,
            'priors': priors,
            'start': start,
            'end': end,
            'loc_feats': loc_feats,
            'conf_feats': conf_feats,
            'out': out
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
