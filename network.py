import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_model_summary import summary

from common.configs import config
from i3d_backbone import InceptionI3d, Unit3D


class _Unit1D(nn.Module):
    def __init__(self, in_channels,
                 output_channels,
                 kernel_shape=1,
                 stride=1,
                 padding='same',
                 activation_fn=F.relu,
                 use_bias=True):
        super(_Unit1D, self).__init__()
        self.conv1d = nn.Conv1d(in_channels,
                                output_channels,
                                kernel_shape,
                                stride,
                                padding=0,
                                bias=use_bias)
        self._activation_fn = activation_fn
        self._padding = padding
        self._stride = stride
        self._kernel_shape = kernel_shape

    def compute_pad(self, t):
        if t % self._stride == 0:
            return max(self._kernel_shape - self._stride, 0)
        else:
            return max(self._kernel_shape - (t % self._stride), 0)

    def forward(self, x):
        if self._padding == 'same':
            batch, channel, t = x.size()
            pad_t = self.compute_pad(t)
            pad_t_f = pad_t // 2
            pad_t_b = pad_t - pad_t_f
            x = F.pad(x, [pad_t_f, pad_t_b])
        x = self.conv1d(x)
        if self._activation_fn is not None:
            x = self._activation_fn(x)
        return x


class _Unit3D(nn.Module):
    def __init__(self, in_channels,
                 output_channels,
                 kernel_shape=(1, 1, 1),
                 stride=(1, 1, 1),
                 padding='spatial_valid',
                 activation_fn=F.relu,
                 use_batch_norm=False,
                 use_bias=False):
        """Initializes Unit3D module."""
        super(_Unit3D, self).__init__()

        self._output_channels = output_channels
        self._kernel_shape = kernel_shape
        self._stride = stride
        self._use_batch_norm = use_batch_norm
        self._activation_fn = activation_fn
        self._use_bias = use_bias
        self.padding = padding

        if self._use_batch_norm:
            self.bn = nn.BatchNorm3d(
                self._output_channels, eps=0.001, momentum=0.01)

        self.conv3d = nn.Conv3d(in_channels=in_channels,
                                out_channels=self._output_channels,
                                kernel_size=self._kernel_shape,
                                stride=self._stride,
                                padding=0,
                                bias=self._use_bias)

    def compute_pad(self, dim, s):
        if s % self._stride[dim] == 0:
            return max(self._kernel_shape[dim] - self._stride[dim], 0)
        else:
            return max(self._kernel_shape[dim] - (s % self._stride[dim]), 0)

    def forward(self, x):
        # compute 'same' padding
        if self.padding == 'same':
            (batch, channel, t, h, w) = x.size()
            pad_t = self.compute_pad(0, t)
            pad_h = self.compute_pad(1, h)
            pad_w = self.compute_pad(2, w)

            pad_t_f = pad_t // 2
            pad_t_b = pad_t - pad_t_f
            pad_h_f = pad_h // 2
            pad_h_b = pad_h - pad_h_f
            pad_w_f = pad_w // 2
            pad_w_b = pad_w - pad_w_f

            pad = [pad_w_f, pad_w_b, pad_h_f, pad_h_b, pad_t_f, pad_t_b]
            x = F.pad(x, pad)

        if self.padding == 'spatial_valid':
            (batch, channel, t, h, w) = x.size()
            pad_t = self.compute_pad(0, t)
            pad_t_f = pad_t // 2
            pad_t_b = pad_t - pad_t_f

            pad = [0, 0, 0, 0, pad_t_f, pad_t_b]
            x = F.pad(x, pad)

        x = self.conv3d(x)
        if self._use_batch_norm:
            x = self.bn(x)
        if self._activation_fn is not None:
            x = self._activation_fn(x)
        return x


class I3D_BackBone(nn.Module):
    def __init__(self, in_channels=3):
        super(I3D_BackBone, self).__init__()
        self._model = InceptionI3d(
            name='inception_i3d', in_channels=in_channels)
        self._model.build()

    def load_pretrained_weight(self, model_path='models/i3d_models/rgb_imagenet.pt'):
        self._model.load_state_dict(torch.load(model_path), strict=False)

    def forward(self, x):
        return self._model.extract_features(x)


class CoarseNetwork(nn.Module):
    def __init__(self, num_classes, in_channels, out_channels):
        super(CoarseNetwork, self).__init__()
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv = nn.Sequential(
            _Unit1D(
                in_channels=self.in_channels,
                output_channels=self.out_channels,
                kernel_shape=3,
                stride=1,
                activation_fn=None,
                use_bias=True
            ),
            nn.GroupNorm(32, self.out_channels),
            nn.ReLU(inplace=True),
            _Unit1D(
                in_channels=self.out_channels,
                output_channels=self.out_channels,
                kernel_shape=3,
                stride=1,
                activation_fn=None,
                use_bias=True
            ),
            nn.GroupNorm(32, self.out_channels),
            nn.ReLU(inplace=True),
            _Unit1D(
                in_channels=self.out_channels,
                output_channels=self.out_channels,
                kernel_shape=1,
                stride=1,
                activation_fn=None,
                use_bias=True
            ),
            nn.GroupNorm(32, self.out_channels),
            nn.ReLU(inplace=True)
        )
        self.local = MLP(input_dim=512, hidden_dim=256,
                         output_dim=2, num_layers=3)
        self.classifier = nn.Linear(512, num_classes + 1)

    def forward(self, x):
        x = self.conv(x)
        x = x.permute(0, 2, 1)
        localized = self.local(x)
        classified = self.classifier(x)

        return {'local': localized, 'class': classified}


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k)
                                    for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)

        return x


class PTN(nn.Module):
    def __init__(self, num_classes, in_channels=3, training=True):
        super(PTN, self).__init__()
        self.num_classes = num_classes
        self.backbone = I3D_BackBone(in_channels=in_channels)
        self._training = training
        self.coarseNet = CoarseNetwork(self.num_classes, 1024, 512)

        if self._training:
            self.backbone.load_pretrained_weight()

    def forward(self, x):
        x = self.backbone(x)
        x = x.squeeze(-1).squeeze(-1)
        coarse_output = self.coarseNet(x)

        return coarse_output


if __name__ == '__main__':
    net = PTN(in_channels=3, training=True)
    input = np.load(
        'datasets/thumos14/test_npy/video_test_0000006.npy')

    input = torch.from_numpy(input).float().permute(
        3, 0, 1, 2).unsqueeze(0)  # 1 x 3 x 670 x 112 x 112
    print(input.shape)

    output = net(input)
    print(output.shape)  # 1 x 512 x 83
