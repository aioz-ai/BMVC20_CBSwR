import torch.optim as optim
from models import build_dual_model
from dataset import MetricLearningDataset
from torch.utils.data import DataLoader
from augmentation import transform_train, transform_test
from torch.autograd import Variable
import math
import torch
import numpy as np
from trainer.trainer import compute_knn
from loss import CenterBatchCriterion, ClusterCriterion, ReconstructCriterion
import matplotlib.pyplot as plt

# deterministic behaviour
torch.manual_seed(1024)
torch.cuda.manual_seed(1024)
torch.backends.cudnn.benchmark = True
np.random.seed(1024)
device = 'cuda:0'
train_set = MetricLearningDataset('data', train=True, transform=transform_train)
trn_loader = DataLoader(train_set, batch_size=32, shuffle=True, num_workers=4, drop_last=True)
model = build_dual_model('default', pretrained=True, low_dim=128, n_cluster=75).to(device)
# state = torch.load('checkpoint/test_val_loss/Sep-27-16_59_12/model_best.15.pth')
state = torch.load('new_checkpoint/test/Sep-28-22_58_44/model_best.12.pth')
model.load_state_dict(state['model'])
optimizer = optim.SGD(model.parameters(), lr=1e-1)
optimizer.load_state_dict(state['optimizer'])
print(optimizer.param_groups[0]['lr'])

ml_criterion = CenterBatchCriterion(1, 0.1, 64, 1)
ml_criterion.to(device)
rim_criterion = ClusterCriterion(1)
rim_criterion.to(device)
recon_criterion = ReconstructCriterion()
recon_criterion.to(device)

def find_lr(init_value = 1e-8, final_value=1e-1, beta = 0.98):
    num = len(trn_loader)-1
    mult = (final_value / init_value) ** (1/num)
    lr = init_value
    optimizer.param_groups[0]['lr'] = lr
    avg_loss = 0.
    best_loss = 0.
    batch_num = 0
    losses = []
    log_lrs = []

    # generate positive index
    train_set = trn_loader.dataset
    n_data = len(train_set)
    temp_loader = torch.utils.data.DataLoader(train_set, batch_size=100, shuffle=False, num_workers=4)
    train_set.transform = transform_test
    labels = np.zeros(n_data)
    model.mode = 'pool'
    out_index = 1
    feat_dim = 1024

    features = torch.zeros(n_data, feat_dim)
    labels = torch.Tensor(labels)
    ptr = 0
    with torch.no_grad():
        for batch_idx, (inputs, _, _) in enumerate(temp_loader):
            batch_size = inputs.size(0)
            real_size = min(batch_size, 100)
            inputs = inputs.to('cuda:0')
            batch_feat = model(inputs)[out_index]
            features[ptr:ptr + real_size, :] = batch_feat.cpu()
            ptr += 100

    train_set.transform = transform_train
    model.mode = 'normal'

    # select nn Index
    dist_feat = np.array(torch.mm(features, features.t()))
    nn_index = compute_knn(dist_feat, labels, knn=1, epoch=1)
    train_set.nnIndex = nn_index

    for inputs1, inputs2, targets in trn_loader:
        batch_num += 1
        #As before, get the loss for this mini-batch of inputs/outputs
        inputs1, inputs2, targets = inputs1.to(device), inputs2.to(device), targets.to(device)
        targets = targets.repeat(2)
        inputs = torch.cat((inputs1, inputs2), 0)
        optimizer.zero_grad()
        repr, cluster, emb = model(inputs)
        # Total loss
        rim_loss = rim_criterion(cluster)
        pred_cluster = torch.argmax(torch.softmax(cluster, dim=1), dim=1)

        unique_cluster = torch.unique(pred_cluster)
        centroid_embedding = torch.zeros(len(unique_cluster), 1024, 7, 7).to(device)
        index = pred_cluster == unique_cluster.view(-1, 1)
        for i in range(len(index)):
            centroid_embedding[i] = torch.mean(emb[index[i]], dim=0)

        emb_index = torch.argmax(unique_cluster == pred_cluster.view(-1, 1), dim=1)
        model.feat_ext.eval()
        x = model.flatten(centroid_embedding.detach().to(device))
        x = model.feat_ext(x)
        centroid_repr = model.l2norm(x)
        model.feat_ext.train()
        ml_loss = ml_criterion(repr, centroid_repr, pred_cluster)
        # metric_loss = self.ml_criterion(repr)
        loss = ml_loss + 0.1 * rim_loss
        # loss = recon_loss + ml_loss + rim_loss
        #Compute the smoothed loss
        avg_loss = beta * avg_loss + (1-beta) * loss.item()
        smoothed_loss = avg_loss / (1 - beta**batch_num)
        #Stop if the loss is exploding
        if batch_num > 1 and smoothed_loss > 4 * best_loss:
            return log_lrs, losses
        #Record the best loss
        if smoothed_loss < best_loss or batch_num==1:
            best_loss = smoothed_loss
        #Store the values
        losses.append(smoothed_loss)
        log_lrs.append(math.log10(lr))
        #Do the SGD step
        loss.backward()
        optimizer.step()
        #Update the lr for the next step
        lr *= mult
        optimizer.param_groups[0]['lr'] = lr
    return log_lrs, losses


logs, losses = find_lr()
plt.plot(logs[10:-5],losses[10:-5])
plt.show()
print(losses)