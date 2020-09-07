import json
import os
from datetime import datetime
import torch


class Config:
    def __init__(self):
        self.cfg = {'_save_dir': None}

    def load_config(self, path):
        self.cfg.update(json.load(open(path, 'r')))

    def save_config(self, path):
        json.dump(self.cfg, open(path, 'w'))

    def update_config(self, args):
        for arg in vars(args):
            self.cfg[arg] = getattr(args, arg)

    def __getattr__(self, item):
        return self.cfg[item]

    def __str__(self):
        msg = '(Arguments: '
        for key, value in self.cfg.items():
            msg += '  {}={}'.format(key, value)
        msg += ')'
        return msg

    @property
    def save_dir(self):
        if self._save_dir:
            return self._save_dir
        else:
            path = self.checkpoint_dir
            make_if_not_exist(path)
            exp_path = os.path.join(path, self.model_name)
            make_if_not_exist(exp_path)
            now = datetime.now()
            time_stamp = now.strftime("%b-%d-%H_%M_%S")
            final_path = os.path.join(exp_path, time_stamp)
            make_if_not_exist(final_path)
            self._save_dir = final_path
            return final_path

    @property
    def device(self):
        return torch.device(self._device)


def make_if_not_exist(path):
    if not os.path.exists(path):
        os.mkdir(path)
