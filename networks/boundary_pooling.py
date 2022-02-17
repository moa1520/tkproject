import torch
import torch.nn as nn


class BoundaryMaxPooling(nn.Module):
    def __init__(self):
        super(BoundaryMaxPooling, self).__init__()

    def forward(self, feature, segments, max_len):
        '''
        feature: B x 2C x T
        segments: B x T x 4
        '''
        c = feature.size(1)
        t = segments.size(1)
        segments_ = torch.clamp(segments, min=0, max=max_len-1)
        start_seg = segments_[:, :, :2].clone()
        end_seg = segments_[:, :, 2:].clone()
        start_boundary_feat = feature[:, :c // 2, :]
        end_boundary_feat = feature[:, c//2:, :]
        new_start_boundary_feat = []
        new_end_boundary_feat = []

        '''
        안되면 이거 써보기
        torch.amax(start_boundary_feat[:, :, torch.floor(
                start_seg[0, i, 0]).int():torch.ceil(start_seg[0, i, 1]).int()], 2)
        '''

        for i in range(t):
            if torch.floor(start_seg[0, i, 0]).int() == torch.ceil(start_seg[0, i, 1]).int():
                # start_seg[0, i, 1] += 1
                new_start_boundary_feat.append(
                    start_boundary_feat[:, :, torch.floor(
                        start_seg[0, i, 0]).int()]
                )
            else:
                new_start_boundary_feat.append(torch.max(start_boundary_feat[:, :, torch.floor(
                    start_seg[0, i, 0]).int():torch.ceil(start_seg[0, i, 1]).int()], dim=-1)[0])
            if torch.floor(end_seg[0, i, 0]).int() == torch.ceil(end_seg[0, i, 1]).int():
                # end_seg[0, i, 0] -= 1
                new_end_boundary_feat.append(
                    end_boundary_feat[:, :, torch.floor(
                        end_seg[0, i, 0]).int()]
                )
            else:
                new_end_boundary_feat.append(torch.max(end_boundary_feat[:, :, torch.floor(
                    end_seg[0, i, 0]).int():torch.ceil(end_seg[0, i, 1]).int()], dim=-1)[0])

        new_start_boundary_feat = torch.stack(new_start_boundary_feat, dim=-1)
        new_end_boundary_feat = torch.stack(new_end_boundary_feat, dim=-1)

        final_boundary_feat = torch.cat(
            [new_start_boundary_feat, new_end_boundary_feat], dim=1)

        return final_boundary_feat
