from models import build_dual_model
import torch
from dataset import MetricLearningDataset
from augmentation import transform_test
from utils import eval_recall_K
from config import Config
import argparse

parser = argparse.ArgumentParser(description='PyTorch: test CBSwR')
parser.add_argument('--test_batch', default=100, type=int,
                    help='training batch size')
parser.add_argument('--low_dim', default=128, type=int,
                    metavar='D', help='feature dimension')
parser.add_argument('--n_cluster', type=int, help='number of cluster', default=32)
parser.add_argument('--checkpoint_path', default='new_checkpoint/CBSwR_CUB200.pth', type=str,
                    help='model save path')
parser.add_argument('--config', type=str, default=None, help='Config location')
parser.add_argument('--dataset', default='cub200', type=str,
                    help='dataset name')

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
    K_list = [1, 2, 4, 8]
    model = build_dual_model('default', True, cfg.low_dim, cfg.n_cluster)
    state = torch.load(cfg.checkpoint_path)
    model.load_state_dict(state['model'])
    model.to(cfg.device)
    dataset = MetricLearningDataset('data', train=False, dataset_name = cfg.dataset, transform=transform_test)


    model.eval()
    model.mode = 'pool'
    n_data = len(dataset)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=cfg.test_batch, shuffle=False, num_workers=4)

    labels = dataset.targets
    out_index = 0

    features = torch.zeros(n_data, cfg.low_dim)
    labels = torch.Tensor(labels)
    ptr = 0
    with torch.no_grad():
        for batch_idx, (inputs, _, _) in enumerate(data_loader):
            batch_size = inputs.size(0)
            real_size = min(batch_size, 100)
            inputs = inputs.to(cfg.device)
            batch_feat = model(inputs)[out_index]
            features[ptr:ptr + real_size, :] = batch_feat.cpu()
            ptr += 100

    print(eval_recall_K(features.numpy(), labels.numpy(), K_list))

