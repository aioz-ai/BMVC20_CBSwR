from torchvision import transforms
from torchvision.utils import make_grid, save_image
import matplotlib.pyplot as plt
from dataset import AutoencoderDataset, MetricLearningDataset
import torch
from torch.utils.data import DataLoader
from models import build_auto_enc_model
import torch.optim as optim
from loss import ReconstructCriterion
from config import Config
import argparse
import numpy as np
import models
from trainer.auto_enc_trainer import AutoEncocderTrainer


class NormalizeInverse(transforms.Normalize):
    """
    Undoes the normalization and returns the reconstructed images in the input domain.
    """

    def __init__(self, mean, std):
        mean = torch.as_tensor(mean)
        std = torch.as_tensor(std)
        std_inv = 1 / (std + 1e-7)
        mean_inv = -mean * std_inv
        super().__init__(mean=mean_inv, std=std_inv)

    def __call__(self, tensor):
        return super().__call__(tensor.clone())


# tra = transforms.Compose([
#     transforms.ToPILImage(),
#     transforms.ToTensor(),
#     transforms.Lambda(lambda x: x * 255.0),
#     transforms.Normalize(mean=[122.7717, 115.9465, 102.9801], std=[1, 1, 1]),
#     transforms.Lambda(lambda x: x[[2, 1, 0], ...])
# ])

inv_tra = transforms.Compose([
    # transforms.Lambda(lambda x: x * 255.0),
    transforms.Lambda(lambda x: x[[2, 1, 0], ...]),
    NormalizeInverse(mean=[122.7717, 115.9465, 102.9801], std=[1., 1., 1.]),
    transforms.Lambda(lambda x: x / 255),
])

# device = torch.device('cuda:0')
# dataset = AutoencoderDataset('data', True)
# dataloader = DataLoader(dataset, batch_size=64, num_workers=4, shuffle=True)
#
# testset = AutoencoderDataset('data', False)
# testloader = DataLoader(testset, batch_size=64, num_workers=4, shuffle=False)
#
# model = build_auto_enc_model('default', True)
# model.to(device)
# optimizer = Adam(model.parameters(), lr=2e-4)
# scheduler = lr_scheduler.StepLR(optimizer, 10)
# criterion = ReconstructCriterion()
# criterion.to(device)
#
#
# for epoch in range(1, 60):
#     for idx, (img, target) in enumerate(dataloader):
#         optimizer.zero_grad()
#         img, target = img.to(device), target.to(device)
#         recon = model(img) * 255.
#         loss = criterion(recon, target)
#         criterion.update(loss.item(), img.size(0))
#         loss.backward()
#         optimizer.step()
#
#         if idx % 20 == 0:
#             print("Epoch {}, Iter [{}|{}], Loss: {}".format(epoch, idx, len(dataloader), criterion.val))
#
#     for img, targets in testloader:
#         for i in range(targets.size(0)):
#             targets[i] = inv_tra(targets[i])
#         grid = make_grid(targets)
#         save_image(grid, 'autoencoder/target_epoch_{}.jpg'.format(epoch))
#
#         img = img.to(device)
#         with torch.no_grad():
#             recon = model(img) * 255.
#
#         recon = recon.detach().cpu()
#         for i in range(targets.size(0)):
#             recon[i] = inv_tra(recon[i])
#         grid = make_grid(recon)
#         save_image(grid, 'autoencoder/recon_epoch_{}.jpg'.format(epoch))
#         break
#
#     scheduler.step()
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch CUB200 AutoEncoder Training')
    parser.add_argument('--arch', default='dual_model', type=str)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--test_batch', default=100, type=int)
    parser.add_argument('--model_name', default='AutoEncoder', type=str, help='Model name')
    parser.add_argument('--checkpoint_dir', default='checkpoint/', type=str,
                        help='model save path')
    parser.add_argument('--config', default=None, type=str)
    parser.add_argument('--seed', default=0, type=int, help='Model name')
    parser.add_argument('--n_epoch', default=40, type=int, help='Model name')



    # ----------------------
    # Setting up
    # ----------------------
    cfg = Config()
    args = parser.parse_args()
    if args.config:
        cfg.load_config(args.config)
    else:
        args._device = "cuda:0" if torch.cuda.is_available() else "cpu"
        cfg.update_config(args)

    # deterministic behaviour
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed(cfg.seed)
    torch.backends.cudnn.benchmark = True
    np.random.seed(cfg.seed)

    # ---------------------------
    # Training
    # ---------------------------
    # train_set = MLDataInstance(src_dir='data', dataset_name='cub200', train=True, transform=transform_train)
    train_set = AutoencoderDataset('data', train=True)
    trainloader = DataLoader(train_set, batch_size=cfg.batch_size, shuffle=True, num_workers=4, drop_last=True)

    # test_set = MLDataInstance(src_dir = 'data', dataset_name='cub200', train=False, transform=transform_test)
    test_set = AutoencoderDataset('data', train=False)
    testloader = DataLoader(test_set, batch_size=cfg.test_batch, shuffle=False, num_workers=4)
    # ----------------
    # Model and Loss
    # ----------------

    # define model
    model = getattr(models, 'build_{}'.format(cfg.arch))(model_type='default', pretrained=True, low_dim=256, n_cluster=100)
    model.to(cfg.device)
    # define optimizer
    optimizer = optim.Adam(model.parameters())
    # scheduler = optim.lr_scheduler.CyclicLR(optimizer, base_lr=0.00017, max_lr=0.001, step_size_up=300, step_size_down=300, mode='triangular2')

    trainer = AutoEncocderTrainer(model, optimizer, cfg, trainloader, testloader)
    trainer.train()
