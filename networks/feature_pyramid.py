import torch
import torch.nn as nn
import torch.nn.functional as F

layer_num = 6
conv_channels = 512
feat_t = 256 // 4


class FPN(nn.Module):
    def __init__(self, num_classes, feature_channels, frame_num=256):
        super(FPN, self).__init__()
        out_channels = conv_channels
        self.num_classes = num_classes
        self.pyramids = nn.ModuleList()
        self.loc_heads = nn.ModuleList()
        self.frame_num = frame_num
        self.layer_num = layer_num
        for i in range(2):
            self.pyramids.append(nn.Sequential(
                _Unit3D(
                    in_channels=feature_channels[i],
                    output_channels=out_channels,
                    kernel_shape=[1, 6 // (i + 1), 6 // (i + 1)],
                    padding='sptial_valid',
                    use_batch_norm=False,
                    use_bias=True,
                    activation_fn=None
                ),
                nn.GroupNorm(32, out_channels),
                nn.ReLU(inplace=True)
            ))
        for i in range(2, layer_num):
            self.pyramids.append(nn.Sequential(
                _Unit1D(
                    in_channels=out_channels,
                    output_channels=out_channels,
                    kernel_shape=3,
                    stride=2,
                    use_bias=True,
                    activation_fn=None
                ),
                nn.GroupNorm(32, out_channels),
                nn.ReLU(inplace=True)
            ))

        loc_block = []
        for i in range(2):
            loc_block.append(nn.Sequential(
                _Unit1D(
                    in_channels=out_channels,
                    output_channels=out_channels,
                    kernel_shape=3,
                    stride=1,
                    use_bias=True,
                    activation_fn=None
                ),
                nn.GroupNorm(32, out_channels),
                nn.ReLU(inplace=True)
            ))
        self.loc_block = nn.Sequential(*loc_block)

        conf_block = []
        for i in range(2):
            conf_block.append(nn.Sequential(
                _Unit1D(
                    in_channels=out_channels,
                    output_channels=out_channels,
                    kernel_shape=3,
                    stride=1,
                    use_bias=True,
                    activation_fn=None
                ),
                nn.GroupNorm(32, out_channels),
                nn.ReLU(inplace=True)
            ))
        self.conf_block = nn.Sequential(*conf_block)

        self.deconv = nn.Sequential(
            _Unit1D(out_channels, out_channels, 3, activation_fn=None),
            nn.GroupNorm(32, out_channels),
            nn.ReLU(inplace=True),
            _Unit1D(out_channels, out_channels, 3, activation_fn=None),
            nn.GroupNorm(32, out_channels),
            nn.ReLU(inplace=True),
            _Unit1D(out_channels, out_channels, 1, activation_fn=None),
            nn.GroupNorm(32, out_channels),
            nn.ReLU(inplace=True),
        )

        self.loc_head = _Unit1D(
            in_channels=out_channels,
            output_channels=2,
            kernel_shape=3,
            stride=1,
            use_bias=True,
            activation_fn=None
        )
        self.conf_head = _Unit1D(
            in_channels=out_channels,
            output_channels=self.num_classes,
            kernel_shape=3,
            stride=1,
            use_bias=True,
            activation_fn=None
        )
        self.center_head = _Unit1D(
            in_channels=out_channels,
            output_channels=1,
            kernel_shape=3,
            stride=1,
            use_bias=True,
            activation_fn=None
        )

        self.priors = []
        t = feat_t
        for i in range(layer_num):
            self.loc_heads.append(ScaleExp())
            self.priors.append(
                torch.Tensor([[(c + 0.5) / t] for c in range(t)]).view(-1, 1)
            )
            t = t // 2

    def forward(self, feat_dict):
        pyramid_feats = []
        locs = []
        confs = []
        centers = []
        loc_feats = []
        conf_feats = []

        x2 = feat_dict['Mixed_5c']
        x1 = feat_dict['Mixed_4f']
        batch_num = x1.size(0)
        for i, conv in enumerate(self.pyramids):
            if i == 0:
                x = conv(x1)
                x = x.squeeze(-1).squeeze(-1)
            elif i == 1:
                x = conv(x2)
                x = x.squeeze(-1).squeeze(-1)
                x0 = pyramid_feats[-1]
                y = F.interpolate(x, x0.size()[2:], mode='nearest')
                pyramid_feats[-1] = x0 + y
            else:
                x = conv(x)
            pyramid_feats.append(x)

        frame_level_feat = pyramid_feats[0].unsqueeze(-1)  # [1, 512, 64, 1]
        frame_level_feat = F.interpolate(
            frame_level_feat, [self.frame_num, 1]).squeeze(-1)  # [1,  512, 256]
        frame_level_feat = self.deconv(frame_level_feat)
        start_feat = frame_level_feat[:, :256]
        end_feat = frame_level_feat[:, 256:]
        start = start_feat.permute(0, 2, 1).contiguous()
        end = end_feat.permute(0, 2, 1).contiguous()

        for i, feat in enumerate(pyramid_feats):
            loc_feat = self.loc_block(feat)
            conf_feat = self.conf_block(feat)
            loc_feats.append(loc_feat)
            conf_feats.append(conf_feat)

            locs.append(
                self.loc_heads[i](self.loc_head(loc_feat))
                .view(batch_num, 2, -1)
                .permute(0, 2, 1).contiguous()
            )
            confs.append(
                self.conf_head(conf_feat).view(batch_num, self.num_classes, -1)
                .permute(0, 2, 1).contiguous()
            )

            t = feat.size(2)
            with torch.no_grad():
                segments = locs[-1] / self.frame_num * t
                priors = self.priors[i].expand(
                    batch_num, t, 1).to(feat.device)  # expand == permute
                new_priors = torch.round(priors * t - 0.5)
                plen = segments[:, :, :1] + segments[:, :, 1:]
                in_plen = torch.clamp(plen / 4.0, min=1.0)
                out_plen = torch.clamp(plen / 10.0, min=1.0)

                l_segment = new_priors - segments[:, :, :1]
                r_segment = new_priors - segments[:, :, 1:]
                segments = torch.cat([
                    torch.round(l_segment - out_plen),
                    torch.round(l_segment + in_plen),
                    torch.round(r_segment - in_plen),
                    torch.round(r_segment + out_plen)
                ], dim=-1)

                decoded_segments = torch.cat(
                    [priors[:, :, :1] * self.frame_num - locs[-1][:, :, :1],
                     priors[:, :, :1] * self.frame_num + locs[-1][:, :, 1:]],
                    dim=-1)
                plen = decoded_segments[:, :, 1:] - \
                    decoded_segments[:, :, :1] + 1.0
                in_plen = torch.clamp(plen / 4.0, min=1.0)
                out_plen = torch.clamp(plen / 10.0, min=1.0)
                frame_segments = torch.cat([
                    torch.round(decoded_segments[:, :, :1] - out_plen),
                    torch.round(decoded_segments[:, :, :1] + in_plen),
                    torch.round(decoded_segments[:, :, 1:] - in_plen),
                    torch.round(decoded_segments[:, :, 1:] + out_plen)
                ], dim=-1)

            centers.append(self.center_head(loc_feat).view(
                batch_num, 1, -1).permute(0, 2, 1).contiguous())

        loc = torch.cat([o.view(batch_num, -1, 2) for o in locs], 1)
        conf = torch.cat([o.view(batch_num, -1, self.num_classes)
                         for o in confs], 1)

        center = torch.cat([o.view(batch_num, -1, 1) for o in centers], 1)
        priors = torch.cat(self.priors, 0).to(loc.device).unsqueeze(0)
        loc_feat = torch.cat([o.permute(0, 2, 1) for o in loc_feats], 1)
        conf_feat = torch.cat([o.permute(0, 2, 1) for o in conf_feats], 1)
        '''
        Segment, Framelevel_segment 아직 안썼음
        '''
        return loc, conf, center, priors, start, end, loc_feat, conf_feat, frame_level_feat, segments, frame_segments


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


class ScaleExp(nn.Module):
    def __init__(self, init_value=1.0):
        super(ScaleExp, self).__init__()
        self.scale = nn.Parameter(torch.FloatTensor([init_value]))

    def forward(self, x):
        return torch.exp(x * self.scale)
