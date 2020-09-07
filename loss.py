import torch.nn as nn
import torch
from torch import nn
from utils import AverageMeter
from utils import entropy


class BaseCriterion(nn.Module):
    _name = 'Base'

    def __init__(self):
        super(BaseCriterion, self).__init__()
        self.device = None
        self.monitor = AverageMeter()

    def to(self, device):
        super(BaseCriterion, self).to(device)
        self.device = device

    def forward(self, *input):
        raise NotImplementedError

    def update(self, value, size):
        self.monitor.update(value, size)

    @property
    def name(self):
        return getattr(self, '_name')

    @property
    def val(self):
        return self.monitor.val

    @property
    def avg(self):
        return self.monitor.avg

    def clear_monitor(self):
        self.monitor = AverageMeter()


class ClusterCriterion(BaseCriterion):
    _name = 'RIM'

    def __init__(self, mu=1):
        super(ClusterCriterion, self).__init__()
        self.mu = mu

    def forward(self, inp):
        p = torch.softmax(inp, dim=1)
        p_ave = torch.sum(p, dim=0) / inp.size(0)
        avg_entropy = entropy(p)
        entropy_avg = entropy(p_ave)
        return avg_entropy + (1 - self.mu * entropy_avg)


class BatchCriterion(BaseCriterion):
    '''
    Compute the loss within each batch
    '''
    _name = 'Metric'

    def __init__(self, neg_m, t, batch_size):
        super(BatchCriterion, self).__init__()
        self.neg_m = neg_m
        self.t = t
        self.diag_mat = 1 - torch.eye(batch_size * 2)

    def forward(self, x):
        batch_size = x.size(0)

        # get positive innerproduct
        reordered_x = torch.cat((x.narrow(0, batch_size // 2, batch_size // 2), \
                                 x.narrow(0, 0, batch_size // 2)), 0)
        # reordered_x = reordered_x.data
        pos = (x * reordered_x.data).sum(1).div_(self.t).exp_()

        # get all innerproduct, remove diag
        all_prob = torch.mm(x, x.t().data).div_(self.t).exp_() * self.diag_mat.to(self.device)
        if self.neg_m == 1:
            all_div = all_prob.sum(1)
        else:
            # remove pos for neg
            all_div = (all_prob.sum(1) - pos) * self.neg_m + pos

        lnPmt = torch.div(pos, all_div)

        # negative probability
        Pon_div = all_div.repeat(batch_size, 1)
        lnPon = torch.div(all_prob, Pon_div.t())
        lnPon = -lnPon.add(-1)

        # equation 7 in ref. A (NCE paper)
        lnPon.log_()
        # also remove the pos term
        lnPon = lnPon.sum(1) - (-lnPmt.add(-1)).log_()
        lnPmt.log_()

        lnPmtsum = lnPmt.sum(0)
        lnPonsum = lnPon.sum(0)

        # negative multiply m
        lnPonsum = lnPonsum * self.neg_m
        loss = - (lnPmtsum + lnPonsum) / batch_size
        return loss


class LossMonitor:
    def __init__(self):
        self.losses = []

    def add_loss(self, loss):
        assert isinstance(loss, BaseCriterion), "Not proper loss"
        self.losses.append(loss)

    def summary(self):
        msg = ''
        for loss in self.losses:
            msg += '{loss.name} Loss: {loss.val:.4f} ({loss.avg:.4f})\t'.format(loss=loss)
        return msg

    def clear(self):
        for loss in self.losses:
            loss.clear_monitor()

    @property
    def results(self):
        results = {}
        for loss in self.losses:
            results[loss.name] = loss.avg
        return results


def create_mask(pred_cluster):
    unique_cluster = torch.unique(pred_cluster)
    n = len(pred_cluster)
    m = len(unique_cluster)
    mask = torch.ones(n, m)
    exp_cluster = unique_cluster.expand(n, m)
    mask[exp_cluster == pred_cluster.view(n, 1)] = 0
    return mask


class CenterBatchCriterion(BaseCriterion):
    '''
    Compute the loss within each batch
    '''
    _name = 'Center'

    def __init__(self, neg_m, t, batch_size, alpha=0.1):
        super(CenterBatchCriterion, self).__init__()
        self.neg_m = neg_m
        self.t = t
        self.diag_mat = 1 - torch.eye(batch_size * 2)
        self.alpha = alpha

    def forward(self, x, centers, targets):
        batch_size = x.size(0)

        # get positive innerproduct
        reordered_x = torch.cat((x.narrow(0, batch_size // 2, batch_size // 2),
                                 x.narrow(0, 0, batch_size // 2)), 0)
        # reordered_x = reordered_x.data

        pos = (x * reordered_x.data).sum(1).div_(self.t).exp_()

        # reordered_x = reordered_x.data
        # pos = (x * reordered_x.data).sum(1).div_(self.t).exp_()

        # get all innerproduct, remove diag
        same_cluster_mask = create_mask(targets).to(self.device)

        # all_prob_pos = torch.mm(x, x.t().data).div_(self.t).exp_() * self.diag_mat.to(self.device)
        all_prob = torch.mm(x, centers.t().data).div_(self.t).exp_()

        if self.neg_m == 1:
            # all_div_pos = all_prob_pos.sum(1)
            all_div = all_prob.sum(1)
        else:
            # remove pos for neg
            # all_div_pos = (all_prob_pos.sum(1) - pos) * self.neg_m + pos
            all_div = (all_prob.sum(1) - pos) * self.neg_m + pos

        lnPmt = torch.div(pos, all_div)
        # negative probability
        Pon_div = all_div.repeat(centers.size(0), 1)
        lnPon = torch.div(all_prob, Pon_div.t())
        lnPon = -lnPon.add(-1)
        # prob of image and its centroid
        _lnPon = lnPon[same_cluster_mask == 0]
        # equation 7 in ref. A (NCE paper)
        lnPon.log_()
        # also remove the pos term
        lnPon = lnPon.sum(1) - _lnPon.log_()
        lnPmt.log_()

        lnPmtsum = lnPmt.sum(0)
        lnPonsum = lnPon.sum(0)

        # negative multiply m
        lnPonsum = lnPonsum * self.neg_m
        loss = - (lnPmtsum + self.alpha * lnPonsum) / batch_size
        return loss


class ReconstructCriterion(BaseCriterion):
    _name = "Recon"

    def __init__(self):
        super(ReconstructCriterion, self).__init__()
        self.criterion = nn.MSELoss()

    def forward(self, inputs, targets):
        return self.criterion(inputs, targets)
