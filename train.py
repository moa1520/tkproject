import os
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from common.configs import config
from common.dataloader import THUMOS_Dataset, get_video_anno, get_video_info
from multisegment_loss import MultiSegmentLoss
from networks.network_testing import PTN

# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # Set the GPU 1 to use


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

resume = config['training']['resume']


def print_training_info():
    print('batch size: ', batch_size)
    print('learning rate: ', learning_rate)
    print('weight decay: ', weight_decay)
    print('max epoch: ', max_epoch)
    print('checkpoint path: ', checkpoint_path)
    print('loc weight: ', config['training']['lw'])
    print('cls weight: ', config['training']['cw'])
    print('piou:', config['training']['piou'])
    print('resume: ', resume)


GLOBAL_SEED = 1


def worker_init_fn(worker_id):
    set_seed(GLOBAL_SEED + worker_id)


def resume_training(resume, model, optimizer):
    start_epoch = 1
    if resume > 0:
        start_epoch += resume
        model_path = os.path.join(
            checkpoint_path, 'checkpoint_{}.pth'.format(resume))
        model.load_state_dict(torch.load(model_path))
        train_path = os.path.join(
            train_state_path, 'checkpoint_{}.pth'.format(resume))
        state_dict = torch.load(train_path)
        optimizer.load_state_dict(state_dict['optimizer'])
        set_rng_state(state_dict['state'])
    return start_epoch


def set_rng_state(states):
    random.setstate(states[0])
    np.random.set_state(states[1])
    torch.set_rng_state(states[2])
    if torch.cuda.is_available():
        torch.cuda.set_rng_state(states[3])


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
    # torch.save(model.module.state_dict(), os.path.join(
    #     checkpoint_path, 'checkpoint_{}.pth'.format(epoch)))
    torch.save(model.state_dict(), os.path.join(
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

    loss_l, loss_c, loss_trans_l, loss_trans_c, loss_ct = CPD_Loss(
        [output['loc'], output['conf'], output['center'], output['priors'][0],
         output['refined_loc'], output['refined_cls']], targets)

    loss_start, loss_end = calc_bce_loss(
        output['start'], output['end'], scores)
    scores_ = F.interpolate(scores, config['training']['num_queries'])
    loss_start_loc_prop, loss_end_loc_prop = calc_bce_loss(output['start_loc_prop'],
                                                           output['end_loc_prop'],
                                                           scores_)
    loss_start_conf_prop, loss_end_conf_prop = calc_bce_loss(output['start_conf_prop'],
                                                             output['end_conf_prop'],
                                                             scores_)
    loss_start = loss_start + 0.1 * \
        (loss_start_loc_prop + loss_start_conf_prop)
    loss_end = loss_end + 0.1 * (loss_end_loc_prop + loss_end_conf_prop)

    return loss_l, loss_c, loss_trans_l, loss_trans_c, loss_start, loss_end, loss_ct


def run_one_epoch(epoch, net, optimizer, data_loader, epoch_step_num, training=True):
    if training:
        net.train()
    else:
        net.eval()

    loss_loc_val = 0
    loss_conf_val = 0
    loss_trans_l_val = 0
    loss_trans_c_val = 0
    loss_ct_val = 0
    loss_start_val = 0
    loss_end_val = 0
    cost_val = 0

    with tqdm(data_loader, total=epoch_step_num, ncols=0) as pbar:
        for n_iter, (clips, targets, scores) in enumerate(pbar):

            loss_l, loss_c, loss_trans_l, loss_trans_c, loss_start, loss_end, loss_ct = one_forward(
                net, clips, targets, scores)

            loss_l = loss_l * config['training']['lw']
            loss_c = loss_c * config['training']['cw']
            loss_trans_l = loss_trans_l * config['training']['lw']
            loss_trans_c = loss_trans_c * config['training']['cw']
            loss_ct = loss_ct * config['training']['cw']
            cost = loss_l + loss_c + loss_trans_l + \
                loss_trans_c + loss_start + loss_end + loss_ct

            optimizer.zero_grad()
            cost.backward()
            optimizer.step()

            loss_loc_val += loss_l.cpu().detach().numpy()
            loss_conf_val += loss_c.cpu().detach().numpy()
            loss_trans_l_val += loss_trans_l.cpu().detach().numpy()
            loss_trans_c_val += loss_trans_c.cpu().detach().numpy()
            loss_ct_val += loss_ct.cpu().detach().numpy()
            loss_start_val += loss_start.cpu().detach().numpy()
            loss_end_val += loss_end.cpu().detach().numpy()
            cost_val += cost.cpu().detach().numpy()
            pbar.set_postfix(loss='{:5f}'.format(
                float(cost.cpu().detach().numpy())))

    loss_loc_val /= (n_iter + 1)
    loss_conf_val /= (n_iter + 1)
    loss_trans_l_val /= (n_iter + 1)
    loss_trans_c_val /= (n_iter + 1)
    loss_ct_val /= (n_iter + 1)
    loss_start_val /= (n_iter + 1)
    loss_end_val /= (n_iter + 1)
    cost_val /= (n_iter + 1)

    save_model(epoch, net, optimizer)

    plog = 'Epoch-{}/{} Loss: Total - {:.5f}, loc - {:.5f}, conf - {:.5f}, ' \
        'trans_loc - {:.5f}, trans_conf - {:.5f}, IoU - {:.5f}, ' \
        'start - {:.5f}, end - {:.5f}'.format(epoch, max_epoch, cost_val, loss_loc_val, loss_conf_val,
                                              loss_trans_l_val, loss_trans_c_val, loss_ct_val, loss_start_val, loss_end_val)
    with open(os.path.join(checkpoint_path, 'training_log.txt'), 'a') as f:
        f.write('================================\n')
        f.write(str(datetime.now()) + '\n')
        f.write(plog + '\n')
        f.write('================================\n')
    print(plog)


if __name__ == '__main__':
    print_training_info()
    set_seed(random_seed)

    '''
    Model
    '''
    net = PTN(num_classes=num_classes, num_queries=config['training']['num_queries'], hidden_dim=config['training']['hidden_dim'],
              in_channels=config['model']['in_channels'], training=True)
    net.cuda()
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
    ????????? ??????
    '''
    # train_data_dict = load_video_data(
    #     train_video_infos, config['dataset']['training']['video_data_path'])
    ''''''
    train_dataset = THUMOS_Dataset(None,
                                   train_video_infos,
                                   train_video_annos)
    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, worker_init_fn=worker_init_fn, pin_memory=True, drop_last=True)
    total_iter_num = len(train_dataset) // batch_size

    '''
    Optimizer
    '''
    optimizer = torch.optim.Adam(net.parameters(),
                                 lr=learning_rate,
                                 weight_decay=weight_decay)
    """
    Start training
    """
    start_epoch = resume_training(resume, net, optimizer)

    for i in range(start_epoch, max_epoch + 1):
        run_one_epoch(i, net=net, optimizer=optimizer, data_loader=train_dataloader,
                      epoch_step_num=len(train_dataset) // batch_size, training=True)
