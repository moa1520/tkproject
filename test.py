import numpy as np

# feat = np.load(
#     '/home/tk/Desktop/ktk/benchmark datasets/thumos14/thumos14-i3d-통합된npy/Thumos14reduced-I3D-JOINTFeatures.npy', allow_pickle=True, encoding='bytes')
video = np.load(
    'datasets/thumos14/validation_npy/video_validation_0000051.npy')
# print(feat[0][0].shape)
video = np.transpose(video, [3, 0, 1, 2])
print(video.shape)
