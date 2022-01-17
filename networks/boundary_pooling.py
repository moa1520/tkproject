import torch
import torch.nn as nn


class BoundaryMaxPooling(nn.Module):
    def __init__(self):
        super(BoundaryMaxPooling, self).__init__()

    def forward(self, feature, segments):
        '''
        feature: B x 2C x T
        segments: B x T x 4
        '''
        c = feature.size(1)
        t = segments.size(1)
        segments = torch.clamp(segments, min=0, max=125)
        start_seg = segments[:, :, :2]
        end_seg = segments[:, :, 2:]
        start_boundary_feat = feature[:, :c // 2, :]
        end_boundary_feat = feature[:, c//2:, :]
        new_start_boundary_feat = []
        new_end_boundary_feat = []

        for i in range(t):
            if torch.floor(start_seg[0, i, 0]).int() == torch.ceil(start_seg[0, i, 1]).int():
                start_seg[0, i, 1] += 1
            new_start_boundary_feat.append(torch.max(start_boundary_feat[:, :, torch.floor(
                start_seg[0, i, 0]).int():torch.ceil(start_seg[0, i, 1]).int()], dim=-1)[0])
            if torch.floor(end_seg[0, i, 0]).int() == torch.ceil(end_seg[0, i, 1]).int():
                end_seg[0, i, 0] -= 1
            new_end_boundary_feat.append(torch.max(end_boundary_feat[:, :, torch.floor(
                end_seg[0, i, 0]).int():torch.ceil(end_seg[0, i, 1]).int()], dim=-1)[0])

        new_start_boundary_feat = torch.stack(new_start_boundary_feat, dim=-1)
        new_end_boundary_feat = torch.stack(new_end_boundary_feat, dim=-1)

        final_boundary_feat = torch.cat(
            [new_start_boundary_feat, new_end_boundary_feat], dim=1)

        return final_boundary_feat
