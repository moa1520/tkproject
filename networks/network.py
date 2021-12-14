import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from common.configs import config
from common.misc import nested_tensor_from_tensor_list
from i3d_backbone import InceptionI3d
from pytorch_model_summary import summary

from networks.feature_pyramid import FPN, MLP, CoarseNetwork
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
            nhead=8,
            num_encoder_layers=2,
            num_decoder_layers=4,
            dim_feedforward=2048,
            dropout=0.1,
            activation='relu',
            normalize_before=True,
            return_intermediate_dec=True)

        self.input_proj = nn.Conv1d(512, hidden_dim, kernel_size=1)
        self.class_embed = nn.Linear(hidden_dim, num_classes)
        self.segments_embed = MLP(hidden_dim, hidden_dim, 2, 3)

        self.query_embed = nn.Embedding(num_queries, hidden_dim)

    def forward(self, x):
        feat = self.backbone(x)
        loc, conf, center, priors, start, end, loc_feats, conf_feats, segments, frame_segments = self.feature_pyramid_net(
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
