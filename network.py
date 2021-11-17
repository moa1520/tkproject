import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_model_summary import summary

from common.configs import config
from i3d_backbone import InceptionI3d


class I3D_BackBone(nn.Module):
    def __init__(self, in_channels=3):
        super(I3D_BackBone, self).__init__()
        self._model = InceptionI3d(final_endpoint='Mixed_5c',
                                   name='inception_i3d', in_channels=in_channels)
        self._model.build()

    def load_pretrained_weight(self, model_path='models/i3d_models/rgb_imagenet.pt'):
        self._model.load_state_dict(torch.load(model_path), strict=False)

    def forward(self, x):
        return self._model.extract_features(x)


class PTN(nn.Module):
    def __init__(self, in_channels=3, training=True):
        super(PTN, self).__init__()

        self.backbone = I3D_BackBone(in_channels=in_channels)
        self._training = training

        if self._training:
            self.backbone.load_pretrained_weight()

    def forward(self, x):
        feature = self.backbone(x)

        return feature


if __name__ == '__main__':
    net = PTN(in_channels=3, training=True).cuda()
    input = torch.Tensor(1, 3,  256, 112, 112).cuda()

    output = net(input)
    print(output.shape)

    # print(summary(net, input))
