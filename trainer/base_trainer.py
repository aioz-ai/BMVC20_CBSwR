import logging
import os
import sys
from datetime import datetime
import torch
from logger import get_logger
from loss import LossMonitor


class BaseTrainer:
    def __init__(self, model, optimizer, cfg):
        self.name = cfg.model_name
        self.n_epoch = cfg.n_epoch
        self.save_dir = cfg.save_dir
        self.start_epoch = 1
        self.model = model
        self.optimizer = optimizer
        self.loss_monitor = LossMonitor()
        self.suffix = self.get_suffix()
        self.logger = self.get_logger()
        self.main_metric = 'accuracy'
        cfg.save_config(os.path.join(self.save_dir, 'config.json'))

    def get_suffix(self):
        time_stamp = datetime.now()
        suffix = "{}_{}".format(self.name, time_stamp.strftime("%b-%d-%H_%M_%S"))
        return suffix

    def step(self, epoch):
        raise NotImplementedError

    def eval(self, epoch):
        raise NotImplementedError

    def train(self):
        best_results = {}
        best_acc = -1e10
        for epoch in range(self.start_epoch, self.n_epoch + 1):
            results = self.step(epoch)
            val_results = self.eval(epoch)
            update_best_results(best_results, val_results)

            self.logger.info('[Epoch] {}'.format(epoch))
            self.log_results('Train', results)
            self.log_results('Eval', val_results)
            self.log_results('Best', best_results)

            if val_results[self.main_metric] > best_acc:
                best_acc = val_results[self.main_metric]
                print('Saving..')
                state = self.get_state_for_save(epoch)
                state.update(results)
                state.update(val_results)
                self.save_checkpoint(state, epoch)

    def save_checkpoint(self, state, epoch):
        checkpoint_file = "model_best.{}.pth".format(epoch)
        torch.save(state, os.path.join(self.save_dir, checkpoint_file))

    def get_logger(self):
        handler = logging.StreamHandler(sys.stdout)
        log_dir = os.path.join(self.save_dir, 'logs')
        if not os.path.exists(log_dir):
            os.mkdir(log_dir)
        # log_file = "{}.txt".format(self.suffix)
        logger = get_logger('train', os.path.join(log_dir, 'info.log'))
        logger.addHandler(handler)
        return logger

    def log_results(self, prefix, results):
        msg = '{} results: '.format(prefix)
        for name, value in results.items():
            msg += '\t{}: {:.4f}'.format(name, value)
        self.logger.info(msg)

    def get_state_for_save(self, epoch):
        state = {
            'model': self.model.state_dict(),
            'epoch': epoch,
            'optimizer': self.optimizer.state_dict(),
        }
        return state



def update_best_results(best, current):
    for key in current.keys():
        if key in best:
            if current[key] > best[key]:
                best[key] = current[key]
        else:
            best[key] = current[key]
