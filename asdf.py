import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from common.dataloader import THUMOS_Dataset, get_video_anno, get_video_info
from common.configs import config

from networks.network import PTN


def main():
    # model = PTN(21, num_queries=126, hidden_dim=512,
    #             in_channels=3, training=True)

    train_video_infos = get_video_info(
        config['dataset']['training']['video_info_path'])
    train_video_annos = get_video_anno(
        train_video_infos, config['dataset']['training']['video_anno_path'])
    train_dataset = THUMOS_Dataset(None,
                                   train_video_infos,
                                   train_video_annos)
    train_dataloader = DataLoader(
        train_dataset, batch_size=1, shuffle=True, pin_memory=True, drop_last=True)

    for n_iter, (clips, targets, scores, ssl_clips, ssl_targets, flags) in enumerate(train_dataloader):
        if n_iter == 2:
            break
        print(scores)

    # input = torch.Tensor(1, 3, 256, 96, 96)
    # output = model(input)

    # print(output['loc'].shape)  # 1 x 126 x 2
    # print(output['conf'].shape)  # 1 x 126 x 21
    # print(output['center'].shape)  # 1 x 126 x 1
    # print(output['priors'].shape)  # 1 x 126 x 1
    # print(output['start'].shape)  # 1 x 256 x 256
    # print(output['end'].shape)  # 1 x 256 x 256
    # print(output['loc_feats'].shape)  # 1 x 126 x 512
    # print(output['conf_feats'].shape)  # 1 x 126 x 512


if __name__ == '__main__':
    main()
