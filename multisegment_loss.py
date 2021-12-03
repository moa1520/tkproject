import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from common.configs import config


def log_sum_exp(x):
    """Utility function for computing log_sum_exp while determining
    This will be used to determine unaveraged confidence loss across
    all examples in a batch.
    Args:
        x (Variable(tensor)): conf_preds from conf layers
    """
    x_max = x.data.max()
    return torch.log(torch.sum(torch.exp(x - x_max), 1, keepdim=True)) + x_max


class FocalLoss_Ori(nn.Module):
    """
    This is a implementation of Focal Loss with smooth label cross entropy supported which is proposed in
    'Focal Loss for Dense Object Detection. (https://arxiv.org/abs/1708.02002)'
        Focal_Loss= -1*alpha*(1-pt)*log(pt)
    :param num_class:
    :param alpha: (tensor) 3D or 4D the scalar factor for this criterion
    :param gamma: (float,double) gamma > 0 reduces the relative loss for well-classified examples (p>0.5) putting more
                    focus on hard misclassified example
    :param smooth: (float,double) smooth value when cross entropy
    :param size_average: (bool, optional) By default, the losses are averaged over each loss element in the batch.
    """

    def __init__(self, num_class, alpha=None, gamma=2, balance_index=-1, size_average=True):
        super(FocalLoss_Ori, self).__init__()
        self.num_class = num_class
        if alpha is None:
            alpha = [0.25, 0.75]
        self.alpha = alpha
        self.gamma = gamma
        self.size_average = size_average
        self.eps = 1e-6

        if isinstance(self.alpha, (list, tuple)):
            assert len(self.alpha) == self.num_class
            self.alpha = torch.Tensor(list(self.alpha))
        elif isinstance(self.alpha, (float, int)):
            assert 0 < self.alpha < 1.0, 'alpha should be in `(0,1)`)'
            assert balance_index > -1
            alpha = torch.ones((self.num_class))
            alpha *= 1 - self.alpha
            alpha[balance_index] = self.alpha
            self.alpha = alpha
        elif isinstance(self.alpha, torch.Tensor):
            self.alpha = self.alpha
        else:
            raise TypeError(
                'Not support alpha type, expect `int|float|list|tuple|torch.Tensor`')

    def forward(self, logit, target):

        if logit.dim() > 2:
            # N,C,d1,d2 -> N,C,m (m=d1*d2*...)
            logit = logit.view(logit.size(0), logit.size(1), -1)
            # [N,C,d1*d2..] -> [N,d1*d2..,C]
            logit = logit.transpose(1, 2).contiguous()
            # [N,d1*d2..,C]-> [N*d1*d2..,C]
            logit = logit.view(-1, logit.size(-1))
        target = target.view(-1, 1).type(torch.int64)  # [N,d1,d2,...]->[N*d1*d2*...,1]

        # -----------legacy way------------
        #  idx = target.cpu().long()
        # one_hot_key = torch.FloatTensor(target.size(0), self.num_class).zero_()
        # one_hot_key = one_hot_key.scatter_(1, idx, 1)
        # if one_hot_key.device != logit.device:
        #     one_hot_key = one_hot_key.to(logit.device)
        # pt = (one_hot_key * logit).sum(1) + epsilon

        # ----------memory saving way--------
        pt = logit.gather(1, target).view(-1) + self.eps  # avoid apply
        logpt = pt.log()

        if self.alpha.device != logpt.device:
            self.alpha = self.alpha.to(logpt.device)

        alpha_class = self.alpha.gather(0, target.view(-1))
        logpt = alpha_class * logpt
        loss = -1 * torch.pow(torch.sub(1.0, pt), self.gamma) * logpt

        if self.size_average:
            loss = loss.mean()
        else:
            loss = loss.sum()
        return loss


def iou_loss(pred, target, weight=None, loss_type='giou', reduction='none'):
    """
    jaccard: A ∩ B / A ∪ B = A ∩ B / (area(A) + area(B) - A ∩ B)
    """
    pred_left = pred[:, 0]
    pred_right = pred[:, 1]
    target_left = target[:, 0]
    target_right = target[:, 1]

    pred_area = pred_left + pred_right
    target_area = target_left + target_right

    eps = torch.finfo(torch.float32).eps

    inter = torch.min(pred_left, target_left) + \
        torch.min(pred_right, target_right)
    area_union = target_area + pred_area - inter
    ious = inter / area_union.clamp(min=eps)

    if loss_type == 'linear_iou':
        loss = 1.0 - ious
    elif loss_type == 'giou':
        ac_uion = torch.max(pred_left, target_left) + \
            torch.max(pred_right, target_right)
        gious = ious - (ac_uion - area_union) / ac_uion.clamp(min=eps)
        loss = 1.0 - gious
    else:
        loss = ious

    if weight is not None:
        loss = loss * weight.view(loss.size())
    if reduction == 'sum':
        loss = loss.sum()
    elif reduction == 'mean':
        loss = loss.mean()
    return loss


def calc_ioa(pred, target):
    pred_left = pred[:, 0]
    pred_right = pred[:, 1]
    target_left = target[:, 0]
    target_right = target[:, 1]

    pred_area = pred_left + pred_right
    eps = torch.finfo(torch.float32).eps

    inter = torch.min(pred_left, target_left) + \
        torch.min(pred_right, target_right)
    ioa = inter / pred_area.clamp(min=eps)
    return ioa


class MultiSegmentLoss(nn.Module):
    def __init__(self, num_classes, overlap_thresh, use_gpu=True, use_focal_loss=False):
        super(MultiSegmentLoss, self).__init__()
        self.num_classes = num_classes
        self.overlap_thresh = overlap_thresh
        self.use_focal_loss = use_focal_loss
        self.use_gpu = use_gpu
        if self.use_focal_loss:
            self.focal_loss = FocalLoss_Ori(
                num_classes, balance_index=0, size_average=False, alpha=0.25)
        self.center_loss = nn.BCEWithLogitsLoss(reduction='sum')

    def forward(self, predictions, targets):
        loc_data, conf_data, center_data, priors = predictions
        num_batch = loc_data.size(0)
        num_priors = priors.size(0)
        num_classes = self.num_classes
        clip_length = config['dataset']['training']['clip_length']
        # match priors and ground truth segments
        loc_t = torch.Tensor(num_batch, num_priors, 2).to(loc_data.device)
        conf_t = torch.Tensor(num_batch, num_priors).to(loc_data.device)

        with torch.no_grad():
            for idx in range(num_batch):
                truths = targets[idx][:, :-1]
                labels = targets[idx][:, -1]
                pre_loc = loc_data[idx]
                '''
                match gt
                '''
                K = priors.size(0)
                N = truths.size(0)
                center = priors[:, 0].unsqueeze(1).expand(K, N)
                left = (
                    center - truths[:, 0].unsqueeze(0).expand(K, N)) * clip_length
                right = (truths[:, 1].unsqueeze(0).expand(
                    K, N) - center) * clip_length
                area = left + right
                maxn = clip_length * 2
                area[left < 0] = maxn
                area[right < 0] = maxn
                best_truth_area, best_truth_idx = area.min(1)

                loc_t[idx][:, 0] = (
                    priors[:, 0] - truths[best_truth_idx, 0]) * clip_length
                loc_t[idx][:, 1] = (
                    truths[best_truth_idx, 1] - priors[:, 0]) * clip_length
                conf = labels[best_truth_idx]
                conf[best_truth_area >= maxn] = 0
                conf_t[idx] = conf

        pos = conf_t > 0  # [num_batch, num_priors]
        pos_idx = pos.unsqueeze(pos.dim()).expand_as(
            loc_data)  # [num_batch, num_priors, 2]
        loc_p = loc_data[pos_idx].view(-1, 2)
        loc_target = loc_t[pos_idx].view(-1, 2)
        if loc_p.numel() > 0:
            loss_l = iou_loss(loc_p, loc_target,
                              loss_type='giou', reduction='sum')
        else:
            loss_l = loc_p.sum()

        # softmax focal loss
        conf_p = conf_data.view(-1, num_classes)
        targets_conf = conf_t.view(-1, 1)
        conf_p = F.softmax(conf_p, dim=1)
        loss_c = self.focal_loss(conf_p, targets_conf)

        N = max(pos.sum(), 1)

        loss_l /= N
        loss_c /= N

        return loss_l, loss_c
