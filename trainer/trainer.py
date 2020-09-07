from augmentation import transform_test, transform_train
import torch
from loss import ClusterCriterion, BatchCriterion, ReconstructCriterion, CenterBatchCriterion
from trainer.base_trainer import BaseTrainer
from utils import eval_recall, eval_nmi, bestMap
from torch.utils.data import DataLoader
import numpy as np
from autoencoder import inv_tra
from torchvision.utils import make_grid, save_image
import os
from utils import AverageMeter
from torch.utils.tensorboard import SummaryWriter


class Trainer(BaseTrainer):
    def __init__(self, model, optimizer, cfg, train_loader, val_loader, scheduler):
        super(Trainer, self).__init__(model, optimizer, cfg)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.main_metric = 'Recall'

        # Reconstruction loss
        self.recon = cfg.recon
        if self.recon:
            self.recon_criterion = ReconstructCriterion()
            self.recon_criterion.to(cfg.device)
            self.loss_monitor.add_loss(self.recon_criterion)

        # RIM setup
        self.rim = cfg.rim
        if self.rim:
            self.rim_criterion = ClusterCriterion(cfg.mu)
            self.rim_criterion.to(cfg.device)
            self.loss_monitor.add_loss(self.rim_criterion)

        # Metric setup
        self.ml = cfg.ml
        if self.ml:
            self.ml_criterion = CenterBatchCriterion(1, 0.1, cfg.batch_size, cfg.alpha)
            # self.ml_criterion = BatchCriterion(1, 0.1, args.batch_size)
            self.ml_criterion.to(cfg.device)
            self.loss_monitor.add_loss(self.ml_criterion)

        self.device = cfg.device
        self.args = cfg
        self.logger.info(cfg)
        self.train_writer = SummaryWriter()
        self.val_writer = SummaryWriter()
        self.scheduler = scheduler

    def step(self, epoch):
        # if epoch > 2:
        #     self.rim = 0
        self.model.train()
        adjust_learning_rate(self.optimizer, epoch, self.args)
        self.loss_monitor.clear()
        # generate positive index
        train_set = self.train_loader.dataset
        train_features, train_targets = extract_features(train_set, self.model, self.args, 'train')
        # select nn Index
        dist_feat = np.array(torch.mm(train_features, train_features.t()))
        nn_index = compute_knn(dist_feat, train_targets, knn=1, epoch=epoch)
        train_set.nnIndex = nn_index

        for batch_idx, (inputs1, inputs2, targets) in enumerate(self.train_loader):
            inputs1, inputs2, targets = inputs1.to(self.device), inputs2.to(self.device), targets.to(self.device)
            targets = targets.repeat(2)
            inputs = torch.cat((inputs1, inputs2), 0)
            self.optimizer.zero_grad()
            repr, cluster, emb = self.model(inputs)
            # Total loss
            loss = 0

            # Compute RIM loss
            if self.rim:
                rim_loss = self.rim_criterion(cluster)
                self.rim_criterion.update(rim_loss.item(), inputs.size(0))
                loss += self.rim * rim_loss

            if self.ml or self.recon:
                pred_cluster = torch.argmax(torch.softmax(cluster, dim=1), dim=1)
                unique_cluster = torch.unique(pred_cluster)
                centroid_embedding = torch.zeros(len(unique_cluster), 1024, 7, 7).to(self.device)
                index = pred_cluster == unique_cluster.view(-1, 1)
                for i in range(len(index)):
                    centroid_embedding[i] = torch.mean(emb[index[i]], dim=0)

                if self.ml:
                    x = self.model.flatten(centroid_embedding.detach().to(self.device))
                    x = self.model.feat_ext(x)
                    centroid_repr = self.model.l2norm(x)
                    metric_loss = self.ml_criterion(repr, centroid_repr, pred_cluster)
                    # metric_loss = self.ml_criterion(repr)
                    self.ml_criterion.update(metric_loss.item(), inputs.size(0))
                    loss += self.ml * metric_loss

                if self.recon:
                    emb_index = torch.argmax(unique_cluster == pred_cluster.view(-1, 1), dim=1)
                    centroid_latent = centroid_embedding[emb_index]
                    recon = self.model.decoder(centroid_latent)
                    recon_loss = self.recon_criterion(recon, inputs / 255.)
                    loss += self.recon * recon_loss
                    self.recon_criterion.update(recon_loss.item(), inputs.size(0))

            # Compute norm loss
            loss.backward()
            # self.scheduler.step()
            self.optimizer.step()
            # if epoch >= 3 and epoch % 3 == 0:
            #     self.optimizer.update_swa()
            #     if epoch >= 6:
            #         self.optimizer.swap_swa_sgd()


            self.train_writer.add_scalar('lr', self.optimizer.param_groups[0]['lr'], epoch * len(self.train_loader) + batch_idx)
            if batch_idx % 20 == 0:
                print('Epoch: [{}][{}/{}]\t'.format(epoch, batch_idx, len(self.train_loader)), end='')
                print(self.loss_monitor.summary())

        self.train_writer.add_scalar('ml_loss', self.ml_criterion.avg, epoch)
        if self.recon:
            self.train_writer.add_scalar('recon', self.recon_criterion.avg, epoch)
        if self.rim:
            self.train_writer.add_scalar('rim', self.rim_criterion.avg, epoch)
        return self.loss_monitor.results

    def eval(self, epoch):
        # with torch.no_grad():
        #     self.model.eval()
        #     rim_loss = AverageMeter()
        #     metric_loss = AverageMeter()
        #     recon_loss = AverageMeter()
        #     val_set = self.val_loader.dataset.transform = transform_train
        #     val_loader = DataLoader(self.val_loader.dataset, shuffle=True, batch_size=64, num_workers=4, drop_last=True)
        #
        #     for batch_idx, (inputs1, inputs2, targets) in enumerate(val_loader):
        #         inputs1, inputs2, targets = inputs1.to(self.device), inputs2.to(self.device), targets.to(self.device)
        #         targets = targets.repeat(2)
        #         inputs = torch.cat((inputs1, inputs2), 0)
        #         repr, cluster, emb = self.model(inputs)
        #         # Total loss
        #         loss = 0
        #
        #         # Compute RIM loss
        #         if self.rim:
        #             rim_loss.update(self.rim_criterion(cluster).item(), inputs.size(0))
        #
        #         if self.ml or self.recon:
        #             pred_cluster = torch.argmax(torch.softmax(cluster, dim=1), dim=1)
        #             unique_cluster = torch.unique(pred_cluster)
        #             centroid_embedding = torch.zeros(len(unique_cluster), 1024, 7, 7).to(self.device)
        #             index = pred_cluster == unique_cluster.view(-1, 1)
        #             for i in range(len(index)):
        #                 centroid_embedding[i] = torch.mean(emb[index[i]], dim=0)
        #
        #             if self.ml:
        #                 x = self.model.flatten(centroid_embedding.detach().to(self.device))
        #                 x = self.model.feat_ext(x)
        #                 centroid_repr = self.model.l2norm(x)
        #                 metric_loss.update(self.ml_criterion(repr, centroid_repr, pred_cluster).item(), inputs.size(0))
        #
        #             if self.recon:
        #                 emb_index = torch.argmax(unique_cluster == pred_cluster.view(-1, 1), dim=1)
        #                 centroid_latent = centroid_embedding[emb_index]
        #                 recon = self.model.decoder(centroid_latent)
        #                 recon_loss.update(self.recon_criterion(recon, inputs/255.).item(), inputs.size(0))
        #
        #     self.val_writer.add_scalar('ml_loss', metric_loss.avg, epoch)
        #     self.val_writer.add_scalar('recon', recon_loss.avg, epoch)
        #     self.val_writer.add_scalar('rim', rim_loss.avg, epoch)
        #     print('Val loss - Metric: {metric.avg:4f} RIM: {rim.avg:4f} Recon: {recon.avg:4f}'.format(metric=metric_loss, rim=rim_loss, recon=recon_loss))
        #     val_set.transform = transform_test
        # testing performance
        print('Extracting features...')
        test_set = self.val_loader.dataset
        test_features, test_targets = extract_features(test_set, self.model, self.args, mode='eval')

        train_set = self.train_loader.dataset
        train_features, train_targets = extract_features(train_set, self.model, self.args, mode='eval')
        recal_train = eval_recall(train_features, train_targets)
        recal = eval_recall(test_features, test_targets)
        nmi = eval_nmi(test_features, test_targets)
        return {'Recall': recal,
                'NMI': nmi,
                'Recall train': recal_train
                # 'Cluster acc': cluster_acc
                }


def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    # if epoch < 10:
    #     lr = args.lr
    # elif epoch >= 10 and epoch < 20:
    #     lr = args.lr * 0.1
    # else:
    #     lr = args.lr * 0.01
    # for param_group in optimizer.param_groups:
    #     param_group['lr'] = lr
    current_lr = optimizer.param_groups[0]['lr']
    if epoch ==7:
        for param_group in optimizer.param_groups:
            param_group['lr'] = current_lr * 10


def compute_knn(dist_feat, targets, knn=5, epoch=8):
    '''
    compute the knn according to instance id/ class id
    '''
    ndata = len(targets)
    nnIndex = np.arange(ndata)

    top_acc = 0.
    # compute the instance knn
    for i in range(ndata):
        dist_feat[i, i] = -1000
        dist_tmp = dist_feat[i, :]
        ind = np.argpartition(dist_tmp, -knn)[-knn:]
        # random 1nn and augmented sample for positive
        nnIndex[i] = np.random.choice([ind[0], i])
    return nnIndex.astype(np.int32)


def set_bn_to_eval(m):
    # 1. no update for running mean and var
    # 2. scale and shift parameters are still trainable
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        m.eval()


def extract_features(dataset, model, args, mode='train'):
    model.eval()
    model.mode = 'pool'
    n_data = len(dataset)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=args.test_batch, shuffle=False, num_workers=4)
    if mode == 'train':
        dataset.transform = transform_test
        labels = np.zeros(n_data)
        out_index = 1
        feat_dim = 1024
    elif mode=='eval':
        labels = dataset.targets
        out_index = 0
        feat_dim = args.low_dim
    else:
        labels = dataset.targets
        out_index = 2
        feat_dim = args.n_cluster

    features = torch.zeros(n_data, feat_dim)
    labels = torch.Tensor(labels)
    ptr = 0
    with torch.no_grad():
        for batch_idx, (inputs, _, _) in enumerate(data_loader):
            batch_size = inputs.size(0)
            real_size = min(batch_size, args.test_batch)
            inputs = inputs.to(args.device)
            batch_feat = model(inputs)[out_index]
            features[ptr:ptr + real_size, :] = batch_feat.cpu()
            ptr += args.test_batch

    if mode == 'train' or mode == 'cluster':
        dataset.transform = transform_train
    model.train()
    model.mode = 'normal'
    return features, labels


