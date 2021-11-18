import os
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, dataloader
from tqdm import tqdm
from common.configs import config
from common.dataloader import (THUMOS_Dataset, get_video_anno, get_video_info,
                               load_video_data)
from network import PTN

batch_size = config['training']['batch_size']
learning_rate = float(config['training']['learning_rate'])
weight_decay = float(config['training']['weight_decay'])
max_epoch = config['training']['max_epoch']
num_classes = config['dataset']['num_classes']
checkpoint_path = config['training']['checkpoint_path']
random_seed = config['training']['random_seed']

train_state_path = os.path.join(checkpoint_path, 'training')
Path(train_state_path).mkdir(exist_ok=True, parents=True)


def print_training_info():
    print('batch size: ', batch_size)
    print('learning rate: ', learning_rate)
    print('weight decay: ', weight_decay)
    print('max epoch: ', max_epoch)
    print('num classes: ', num_classes)
    print('checkpoint path: ', checkpoint_path)


def get_rng_states():
    states = []
    states.append(random.getstate())
    states.append(np.random.get_state())
    states.append(torch.get_rng_state())
    if torch.cuda.is_available():
        states.append(torch.cuda.get_rng_state())
    return states


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def save_model(epoch, model, optimizer):
    torch.save(model.module.state_dict(), os.path.join(
        checkpoint_path, 'checkpoint_{}.pth'.format(epoch)))
    torch.save({'optimizer': optimizer.state_dict(),
                'state': get_rng_states()},
               os.path.join(train_state_path, 'checkpoint_{}.pth'.format(epoch)))


def one_forward(net, clips, targets):
    clips = clips.cuda()
    targets = [t.cuda() for t in targets]

    output = net(clips)

    return output


def run_one_epoch(epoch, net, optimizer, data_loader, epoch_step_num, training=True):
    if training:
        net.train()
    else:
        net.eval()

    with tqdm(data_loader, total=epoch_step_num, ncols=0) as pbar:
        for iter, (clips, targets, scores, ssl_clips, ssl_targets, flags) in enumerate(pbar):
            net.cuda()

            output = one_forward(net, clips, targets)
            local, cls = output['local'], output['class']
            target_start = targets[:, :, 0]
            target_end = targets[:, :, 1]
            target_class = targets[:, :, 2]

            print(target_start)
            print(target_end)
            print(target_class)

            if iter == 3:
                break


if __name__ == '__main__':
    print_training_info()
    set_seed(random_seed)

    '''
    Model
    '''
    net = PTN(num_classes=num_classes,
              in_channels=config['model']['in_channels'], training=True)
    '''
    Dataloader
    '''
    train_video_infos = get_video_info(
        config['dataset']['training']['video_info_path'])
    train_video_annos = get_video_anno(
        train_video_infos, config['dataset']['training']['video_anno_path'])
    '''
    이부분 삭제
    '''
    # train_data_dict = load_video_data(
    #     train_video_infos, config['dataset']['training']['video_data_path'])
    ''''''
    train_dataset = THUMOS_Dataset(None,
                                   train_video_infos,
                                   train_video_annos)
    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, drop_last=True)
    total_iter_num = len(train_dataset) // batch_size

    '''
    Optimizer
    '''
    optimizer = torch.optim.Adam(net.parameters(),
                                 lr=learning_rate,
                                 weight_decay=weight_decay)
    '''
    Training
    '''
    run_one_epoch(epoch=0, net=net, optimizer=optimizer, data_loader=train_dataloader,
                  epoch_step_num=len(train_dataset) // batch_size, training=True)
