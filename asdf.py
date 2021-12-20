from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from common import videotransforms
from common.configs import config
from common.dataloader_flow_fusion import (THUMOS_Dataset, get_video_anno,
                                           get_video_info)
from networks.network import PTN
from multisegment_loss_flow_fusion import MultiSegmentLoss


def main():
    model = PTN(21, num_queries=126, hidden_dim=256)
    model.eval()
    model.cuda()

    train_video_infos = get_video_info(
        config['dataset']['training']['video_info_path'])
    train_video_annos = get_video_anno(
        train_video_infos, config['dataset']['training']['video_anno_path'])
    train_dataset = THUMOS_Dataset(None,
                                   train_video_infos,
                                   train_video_annos)
    train_dataloader = DataLoader(
        train_dataset, batch_size=1, shuffle=True, pin_memory=True, drop_last=True)

    MSLoss = MultiSegmentLoss(
        21, config['training']['piou'], use_focal_loss=config['training']['focal_loss'])

    for n_iter, (clips, flow_clips, targets, scores) in enumerate(train_dataloader):
        clips, flow_clips, targets = clips.cuda(), flow_clips.cuda(), targets.cuda()
        print(clips.shape)
        print(flow_clips.shape)
        print(targets.shape)

        out = model(clips, flow_clips)

        print(out['coarse_logits'].shape, out['coarse_segments'].shape,
              out['refine_logits'].shape, out['refine_segments'].shape)

        MSLoss([out['coarse_segments'], out['coarse_logits'],
               out['refine_segments'][0], out['refine_logits'][0]], targets)

        if n_iter == 2:
            break


if __name__ == '__main__':
    # main()

    cetner_crop = videotransforms.CenterCrop(size=96)

    # rgb = np.load('datasets/thumos14/test_npy/video_test_0000051.npy')
    # flow = np.load('datasets/thumos14/test_flow_npy/video_test_0000051.npy')

    # rgb = np.transpose(rgb, [3, 0, 1, 2])
    # rgb = cetner_crop(rgb)
    # rgb = torch.from_numpy(rgb).float().unsqueeze(0)
    # rgb = (rgb / 255.0) * 2.0 - 1.0

    # flow = np.transpose(flow, [3, 0, 1, 2])
    # flow = cetner_crop(flow)
    # flow = torch.from_numpy(flow).float().unsqueeze(0)
    # flow = (flow / 255.0) * 2.0 - 1.0

    # print(flow.shape)
    rgb = torch.Tensor(1, 3, 256, 96, 96)
    print(rgb.shape)

    net = PTN(num_classes=21, num_queries=126, hidden_dim=256)

    out = net(rgb)
    print(out.shape)
