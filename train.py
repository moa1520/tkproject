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
from multisegment_loss import MultiSegmentLoss
from network import PTN
import torch.nn.functional as F

batch_size = config['training']['batch_size']
learning_rate = float(config['training']['learning_rate'])
weight_decay = float(config['training']['weight_decay'])
max_epoch = config['training']['max_epoch']
num_classes = config['dataset']['num_classes']
checkpoint_path = config['training']['checkpoint_path']
random_seed = config['training']['random_seed']
focal_loss = config['training']['focal_loss']

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


def calc_bce_loss(start, end, scores):
    start = torch.tanh(start).mean(-1)
    end = torch.tanh(end).mean(-1)
    loss_start = F.binary_cross_entropy(start.view(-1),
                                        scores[:, 0].contiguous(
    ).view(-1).cuda(),
        reduction='mean')
    loss_end = F.binary_cross_entropy(end.view(-1),
                                      scores[:, 1].contiguous(
    ).view(-1).cuda(),
        reduction='mean')
    return loss_start, loss_end


def one_forward(net, clips, targets, scores=None, training=True):
    clips = clips.cuda()
    targets = [t.cuda() for t in targets]

    if training:
        output = net(clips)
    else:
        with torch.no_grad():
            output = net(clips)

    loss_l, loss_c = CPD_Loss(
        [output['loc'], output['conf'], output['center'], output['priors'][0]], targets)

    loss_start, loss_end = calc_bce_loss(
        output['start'], output['end'], scores)

    return loss_l, loss_c


def run_one_epoch(epoch, net, optimizer, data_loader, epoch_step_num, training=True):
    net.cuda()
    if training:
        net.train()
    else:
        net.eval()

    loss_loc_val = 0
    loss_conf_val = 0
    loss_prop_l_val = 0
    loss_prop_c_val = 0
    loss_ct_val = 0
    loss_start_val = 0
    loss_end_val = 0
    loss_trip_val = 0
    loss_contras_val = 0
    cost_val = 0

    with tqdm(data_loader, total=epoch_step_num, ncols=0) as pbar:
        for iter, (clips, targets, scores, ssl_clips, ssl_targets, flags) in enumerate(pbar):

            loss_l, loss_c = one_forward(net, clips, targets, scores)

            if iter == 0:
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
    Setup loss
    '''
    piou = config['training']['piou']
    CPD_Loss = MultiSegmentLoss(num_classes, piou, use_focal_loss=focal_loss)
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
