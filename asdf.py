# from datetime import datetime

# import numpy as np
import torch
# import torch.nn as nn
from torch.utils.data import DataLoader

# from common import videotransforms
from common.configs import config
from common.dataloader import THUMOS_Dataset, get_video_anno, get_video_info
from networks.network import PTN
from multisegment_loss import MultiSegmentLoss


def main():
    model = PTN(21, num_queries=126, hidden_dim=512, training=False)
    model.eval()

    train_video_infos = get_video_info(
        config['dataset']['testing']['video_info_path'])
    train_video_annos = get_video_anno(
        train_video_infos, config['dataset']['testing']['video_anno_path'])
    train_dataset = THUMOS_Dataset(None,
                                   train_video_infos,
                                   train_video_annos)
    train_dataloader = DataLoader(
        train_dataset, batch_size=1, shuffle=False, pin_memory=True, drop_last=True)

    MSLoss = MultiSegmentLoss(
        21, config['training']['piou'], use_focal_loss=config['training']['focal_loss'])

    for n_iter, (clips, targets, scores) in enumerate(train_dataloader):
        clips, targets = clips, targets
        print(clips.shape)
        print(targets.shape)

        output = model(clips)

        print(output['loc'].shape, output['conf'].shape,
              output['refined_loc'].shape, output['refined_cls'].shape)

        MSLoss([output['loc'], output['conf'], output['center'], output['priors']
               [0], output['refined_loc'], output['refined_cls']], targets)

        if n_iter == 2:
            break


if __name__ == '__main__':
    main()

    # cetner_crop = videotransforms.CenterCrop(size=96)

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
    # rgb = torch.Tensor(1, 3, 256, 96, 96).cuda()
    # print(rgb.shape)

    # net = PTN(num_classes=21, num_queries=126, hidden_dim=256).cuda()

    # out = net(rgb)
    # print(out.shape)
