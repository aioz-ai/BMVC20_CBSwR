import torch
import argparse
from torch import optim
from dataset import MetricLearningDataset
import numpy as np
import models, utils
from torch.utils.data import DataLoader
from config import Config
from augmentation import transform_train, transform_test
from models import build_dual_model
from utils import AverageMeter, entropy
from utils import eval_recall, eval_nmi, eval_recall_numpy
from logger import get_logger
import logging
import sys
import os
from torch.nn import functional
import time
from torchvision import transforms



parser = argparse.ArgumentParser(description='PyTorch: train CBSwR')
parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
parser.add_argument('--t', default=0.1, type=float,
                    metavar='T', help='temperature parameter for softmax')
parser.add_argument('--batch_m', default=1, type=int,
                    metavar='N', help='m for negative sum')
parser.add_argument('--test_batch', default=100, type=int,
                    help='training batch size')
parser.add_argument('--batch_size', default=64, type=int,
                    metavar='B', help='training batch size')
parser.add_argument('--low_dim', default=128, type=int,
                    metavar='D', help='feature dimension')
parser.add_argument('--mu', type=float, help='trade-off parameter for entropy minimization and entropy maximization',
                    default=1)
parser.add_argument('--alpha', type=float, help='weight of second term in batch loss', default=1.0)

parser.add_argument('--rim', type=float, help='weight of RIM loss', default=0.3)
parser.add_argument('--recon', type=float, help='weight of Recon loss', default=0.001)

parser.add_argument('--norm', type=float, help='weight of norm loss', default=0.001)
parser.add_argument('--cl', type=float, help='weight of center loss', default=0.001)
parser.add_argument('--ml', type=float, help='weight of metric learning loss', default=0.9)
parser.add_argument('--n_epoch', type=int, help='number of epoch', default=30)
parser.add_argument('--interval', type=int, help='number of saved epoch', default=5)
parser.add_argument('--n_cluster', type=int, help='number of cluster', default=100)
parser.add_argument('--log_dir', default='log/', type=str,
                    help='log save path')
parser.add_argument('--model_name', default='CBSwR_CUB200', type=str,
                    help='log save path')
parser.add_argument('--checkpoint_dir', default='new_checkpoint/', type=str,
                    help='model save path')
parser.add_argument('--resume', type=str, default=None, help='Checkpoint location')
parser.add_argument('--config', type=str, default=None, help='Config location')
parser.add_argument('--seed', type=int, help='random seed', default=1024)
parser.add_argument('--neg_m', type=int, help='criterion', default=1)
parser.add_argument('--dataset', default='cub200', type=str,
                    help='dataset name')


# -----------------
# Helper function
# -----------------

def compute_knn(dist_feat, targets, knn=5):
    '''
    compute the knn according to instance id/ class id
    '''
    ndata = len(targets)
    nnIndex = np.arange(ndata)
    # compute the instance knn
    for i in range(ndata):
        dist_feat[i, i] = -1000
        dist_tmp = dist_feat[i, :]
        ind = np.argpartition(dist_tmp, -knn)[-knn:]
        # random 1nn and augmented sample for positive
        nnIndex[i] = np.random.choice([ind[0], i])
    return nnIndex.astype(np.int32)

def extract_features(model, dataset):
    n_data = len(dataset)
    feat_dim = cfg.low_dim
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=cfg.test_batch, shuffle=False, num_workers=4)
    model.eval()
    model.mode = 'pool'
    # Extract features
    features = torch.zeros(n_data, feat_dim)
    targets = dataset.targets
    ptr = 0
    with torch.no_grad():
        for batch_idx, (inputs, _, _) in enumerate(data_loader):
            batch_size = inputs.size(0)
            real_size = min(batch_size, args.test_batch)
            inputs = inputs.to(cfg.device)
            repr, _, _ = model(inputs)
            features[ptr:ptr + real_size, :] = repr.cpu()
            ptr += cfg.test_batch
    model.mode = 'normal'
    model.train()
    return features, targets


def get_nearest_idex(model, dataset):
    n_data = len(dataset)
    feat_dim = 1024
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=cfg.test_batch, shuffle=False, num_workers=4)
    model.eval()
    model.mode = 'pool'
    dataset.transform = transform_test
    # Extract features
    print('Extracting features ...')
    features = torch.zeros(n_data, feat_dim)
    targets = torch.zeros(n_data)
    ptr = 0
    with torch.no_grad():
        for batch_idx, (inputs, _, _) in enumerate(data_loader):
            batch_size = inputs.size(0)
            real_size = min(batch_size, args.test_batch)
            inputs = inputs.to(cfg.device)
            _, emb, _ = model(inputs)
            features[ptr:ptr + real_size, :] = emb.cpu()
            ptr += cfg.test_batch
    model.mode = 'normal'
    model.train()
    dataset.transform = transform_train
    # select nn Index
    dist_feat = np.array(torch.mm(features, features.t()))
    nn_index = compute_knn(dist_feat, targets, knn=1)
    return nn_index


def create_mask(pred_cluster):
    unique_cluster = torch.unique(pred_cluster)
    n = len(pred_cluster)
    m = len(unique_cluster)
    mask = torch.ones(n, m)
    exp_cluster = unique_cluster.expand(n, m)
    mask[exp_cluster == pred_cluster.view(n, 1)] = 0
    return mask


def rim_criterion(inp):
    p = torch.softmax(inp, dim=1)
    p_ave = torch.sum(p, dim=0) / inp.size(0)
    avg_entropy = entropy(p)
    entropy_avg = entropy(p_ave)
    return avg_entropy + (1 - cfg.mu * entropy_avg)


def center_batch_criterion(x, centers, targets):
    batch_size = x.size(0)

    reordered_x = torch.cat((x.narrow(0, batch_size // 2, batch_size // 2),
                             x.narrow(0, 0, batch_size // 2)), 0)

    pos = (x * reordered_x.data).sum(1).div_(cfg.t).exp_()

    same_cluster_mask = create_mask(targets).to(cfg.device)

    all_prob = torch.mm(x, centers.t().data).div_(cfg.t).exp_()

    if cfg.neg_m == 1:
        all_div = all_prob.sum(1)
        all_div_pos = (all_prob * same_cluster_mask).sum(1)
    else:
        all_div = (all_prob.sum(1) - pos) * cfg.neg_m + pos

    lnPmt = torch.div(pos, all_div_pos)
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
    lnPonsum = lnPonsum * cfg.neg_m
    loss = - (lnPmtsum + cfg.alpha * lnPonsum) / batch_size
    return loss


def recon_criterion(target, gt):
    return functional.mse_loss(target, gt)


if __name__ == "__main__":
    # ----------------------
    # Setting up
    # ----------------------
    cfg = Config()
    args = parser.parse_args()
    if args.config:
        cfg.load_config(args.config)
    else:
        args.pool_dim = args.low_dim
        args._device = "cuda:0" if torch.cuda.is_available() else "cpu"
        cfg.update_config(args)

    # deterministic behaviour
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed(cfg.seed)
    torch.backends.cudnn.benchmark = True
    np.random.seed(cfg.seed)

    # ----------------------
    # Prepare dataset
    # ----------------------

    train_set = MetricLearningDataset('data', train=True, dataset_name=cfg.dataset, transform=transform_train)
    train_loader = DataLoader(train_set, batch_size=cfg.batch_size, shuffle=True, num_workers=4, drop_last=True)

    test_set = MetricLearningDataset('data', train=False, dataset_name=cfg.dataset, transform=transform_test)
    test_loader = DataLoader(test_set, batch_size=cfg.test_batch, shuffle=False, num_workers=4)


    # ----------------
    # Model and Loss
    # ----------------

    # define model
    model = build_dual_model('default', True, cfg.low_dim, cfg.n_cluster)
    model.to(cfg.device)
    # define optimizer
    optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=cfg.lr)

    # define logger
    handler = logging.StreamHandler(sys.stdout)
    log_dir = os.path.join(cfg.save_dir, 'logs')
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    logger = get_logger('train', os.path.join(log_dir, 'info.log'))
    logger.addHandler(handler)
    logger.info(cfg)
    # Training
    start_epoch = 1
    best_recall = 0
    train_time_total = 0.0
    for epoch in range(start_epoch, cfg.n_epoch + 1):
        # if epoch > 1:
        #     cfg.rim = 0
        train_time_start = time.time()
        rim_loss_mnt = AverageMeter()
        ml_loss_mnt = AverageMeter()
        recon_loss_mnt = AverageMeter()
        if cfg.dataset == 'cub200':
            nn_index = get_nearest_idex(model, train_set)
            train_set.nnIndex = nn_index
        model.train()
        for batch_idx, (inputs1, inputs2, targets) in enumerate(train_loader):
            inputs1, inputs2, targets = inputs1.to(cfg.device), inputs2.to(cfg.device), targets.to(cfg.device)
            targets = targets.repeat(2)
            inputs = torch.cat((inputs1, inputs2), 0)


            optimizer.zero_grad()
            repr, cluster, emb = model(inputs)
            # Total loss
            metric_loss = 0
            recon_loss = 0
            rim_loss = 0

            # Compute RIM loss
            if cfg.rim:
                rim_loss = rim_criterion(cluster)
                rim_loss_mnt.update(rim_loss.item(), inputs.size(0))
                # loss += cfg.rim * rim_loss

            if cfg.ml or cfg.recon:
                pred_cluster = torch.argmax(torch.softmax(cluster, dim=1), dim=1)
                pred_cluster = pred_cluster[:cfg.batch_size]
                pred_cluster = pred_cluster.repeat(2)
                # Uncomment the below line for training with supervised cluster
                # pred_cluster = targets
                unique_cluster = torch.unique(pred_cluster)
                centroid_embedding = torch.zeros(len(unique_cluster), 1024, 7, 7).to(cfg.device)
                index = pred_cluster == unique_cluster.view(-1, 1)
                for i in range(len(index)):
                    centroid_embedding[i] = torch.mean(emb[index[i]], dim=0)

                if cfg.ml:
                    x = model.flatten(centroid_embedding.detach().to(cfg.device))
                    model.feat_ext.eval()
                    x = model.feat_ext(x)
                    centroid_repr = model.l2norm(x)
                    model.feat_ext.train()
                    metric_loss = center_batch_criterion(repr, centroid_repr, pred_cluster)
                    ml_loss_mnt.update(metric_loss.item(), inputs.size(0))
                    # loss += cfg.ml * metric_loss

                if cfg.recon:
                    emb_index = torch.argmax(unique_cluster == pred_cluster.view(-1, 1), dim=1)
                    centroid_latent = centroid_embedding[emb_index]
                    recon = model.decoder(centroid_latent)
                    recon_loss = recon_criterion(recon, inputs / 255.)
                    recon_loss_mnt.update(recon_loss.item(), inputs.size(0))
                    # loss += cfg.recon * recon_loss
            loss = cfg.ml * metric_loss + cfg.recon * recon_loss + cfg.rim * rim_loss
            # Compute norm loss
            loss.backward()
            optimizer.step()

            if batch_idx % 20 == 0:
                print('Epoch: [{}][{}/{}]\t'.format(epoch, batch_idx, len(train_loader)), end='')
                print('Metric loss: {metric.val:4f} ({metric.avg:4f})\t'
                      'Rim loss: {rim.val:.4f} ({rim.avg:.4f})\t'
                      'Recon loss: {recon.val:.4f} ({recon.avg:.4f})'.format(metric=ml_loss_mnt, rim=rim_loss_mnt, recon=recon_loss_mnt))
                # print('lr {}'.format(optimizer.param_groups[0]['lr']))
        train_time_end = time.time()
        train_time_epoch = train_time_end - train_time_start
        logger.info('Training time: {:.6f}'.format(train_time_epoch))
        train_time_total += train_time_epoch

        # Evaluate
        print('Learning rate at epoch {} is {}'.format(epoch, optimizer.param_groups[0]['lr']))
        print('Extracting features...')
        test_features, test_targets = extract_features(model, test_set)
        train_recall = 0
        test_recall = eval_recall_numpy(test_features, test_targets)
        if cfg.dataset == 'ebay':
            nmi = 0.0
        else:
            nmi = eval_nmi(test_features, test_targets)
        if test_recall > best_recall:
            best_recall = test_recall
            # save checkpoint
            state = {
                'model': model.state_dict(),
                'epoch': epoch,
                'optimizer': optimizer.state_dict(),
            }
            checkpoint_file = "model_best.{}.pth".format(epoch)
            torch.save(state, os.path.join(cfg.save_dir, checkpoint_file))
        logger.info('Epoch {}'.format(epoch))
        logger.info('Train Recall: {:.6f}, Test Recall: {:.6f}, NMI: {:.6f}'.format(train_recall, test_recall, nmi))
        logger.info('Best Recall: {:.6f}'.format(best_recall))
    logger.info('Training time total: {:.6f}'.format(train_time_total))
