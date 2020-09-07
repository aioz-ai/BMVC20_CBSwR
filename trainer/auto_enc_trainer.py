from utils import inv_tra
from .base_trainer import BaseTrainer
from loss import ReconstructCriterion
import torch
from torchvision.utils import make_grid, save_image
import os


class AutoEncocderTrainer(BaseTrainer):
    def __init__(self, model, optimizer, cfg, train_loader, val_loader):
        super(AutoEncocderTrainer, self).__init__(model, optimizer, cfg)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.main_metric = 'MSE'
        self.device = cfg.device
        self.args = cfg
        self.logger.info(cfg)
        self.criterion = ReconstructCriterion()
        self.loss_monitor.add_loss(self.criterion)

    def step(self, epoch):
        self.model.train()
        for batch_idx, (img, target) in enumerate(self.train_loader):
            self.optimizer.zero_grad()
            img, target = img.to(self.device), target.to(self.device)
            _, _, emb = self.model(img)
            recon = self.model.decoder(emb)
            loss = self.criterion(recon, target / 255.)
            self.criterion.update(loss.item(), img.size(0))
            loss.backward()
            # self.scheduler.step()
            self.optimizer.step()

            if batch_idx % 20 == 0:
                print('Epoch: [{}][{}/{}]\t'.format(epoch, batch_idx, len(self.train_loader)), end='')
                print(self.loss_monitor.summary())
        return self.loss_monitor.results

    def eval(self, epoch):
        self.model.eval()
        def eval_mean_square(input, target):
            return ((input - target) ** 2).mean()
        metric = []
        for batch_idx, (img, target) in enumerate(self.val_loader):
            img, target = img.to(self.device), target.to(self.device)
            with torch.no_grad():
                _, _, emb = self.model(img)
                recon = self.model.decoder(emb) * 255
                score = eval_mean_square(recon.detach().cpu(), target.detach().cpu())
                metric.append(score)

            if batch_idx == 0:
                recon = recon.detach().cpu()
                for i in range(recon.size(0)):
                    recon[i] = inv_tra(recon[i])
                grid = make_grid(recon, nrow=10)
                save_image(grid, os.path.join(self.save_dir, 'recon_epoch_{}.jpg'.format(epoch)))
        return {'MSE': -sum(metric) / len(metric)}

    def get_state_for_save(self, epoch):
        state = {
            'encoder': self.model.state_dict(),
            'epoch': epoch,
            'optimizer': self.optimizer.state_dict(),
        }
        return state


