import torch
import torch.nn as nn
import numpy as np

from i3d_backbone import InceptionI3d


def run(file):
    model = InceptionI3d(400, in_channels=3)
    model.load_state_dict(torch.load('models/i3d_models/rgb_imagenet.pt'))
    model.cuda()
    file = file.cuda()

    features = model.extract_features(file)

    print(features['Mixed_5c'].shape)


if __name__ == '__main__':
    file_path = 'datasets/thumos14/test_npy/video_test_0000004.npy'
    file = np.load(file_path)
    file = torch.from_numpy(file).float().permute(3, 0, 1, 2).unsqueeze(0)
    b, c, t, h, w = file.shape
    print(b, c, t, h, w)

    run(file)
