import os
import numpy as np
import torch
from sklearn.metrics import accuracy_score
from sklearn.utils import linear_assignment_
from torchvision import transforms
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import normalized_mutual_info_score


def get_data(root, train=True):
    if train:
        split = 'train.txt'
    else:
        split = 'test.txt'

    split_path = os.path.join(root, split)
    imgs = []
    labels = []
    cls_to_id = []
    with open(split_path, 'r') as f:
        for line in f:
            img_path = os.path.join(root, 'images', line.strip())
            cls_name = os.path.split(line)[0]
            if cls_name in cls_to_id:
                target = cls_to_id.index(cls_name)
            else:
                target = len(cls_to_id)
                cls_to_id.append(cls_name)
            imgs.append(img_path)
            labels.append(target)
    return imgs, labels


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def eval_recall_numpy(embedding, label):
    embedding = np.asarray(embedding)
    label = np.asarray(label)
    norm = np.sum(embedding*embedding,axis = 1)
    right_num = 0
    for i in range(embedding.shape[0]):
        dis = norm[i] + norm - 2*np.squeeze(np.matmul(embedding[i],embedding.T))
        dis[i] = 1e10
        pred = np.argmin(dis)
        if label[i]==label[pred]:
            right_num = right_num+1
    recall = float(right_num)/float(embedding.shape[0])
    return recall

def eval_recall(embedding, label):
    norm = torch.sum(embedding * embedding, dim=1)
    right_num = 0
    for i in range(embedding.size(0)):
        dis = norm[i] + norm - 2*np.squeeze(np.matmul(embedding[i],embedding.t()))
        dis[i] = 1e10
        pred = torch.argmin(dis)
        if label[i]==label[pred]:
            right_num = right_num+1
    recall = float(right_num)/float(embedding.size(0))
    return recall


# def eval_recall(embedding, label):
#     norm = np.sum(embedding*embedding,axis = 1)
#     right_num = 0
#     for i in range(embedding.shape[0]):
#         dis = norm[i] + norm - 2*np.squeeze(np.matmul(embedding[i],embedding.T))
#         dis[i] = 1e10
#         pred = np.argmin(dis)
#         if label[i]==label[pred]:
#             right_num = right_num+1
#     recall = float(right_num)/float(embedding.shape[0])
#     return recall


def eval_nmi(embedding, label,  normed_flag = False, fast_kmeans = False):
    unique_id = np.unique(label)
    num_category = len(unique_id)
    if normed_flag:
        for i in range(embedding.shape[0]):
            embedding[i,:] = embedding[i,:]/np.sqrt(np.sum(embedding[i,:] ** 2)+1e-4)
    if fast_kmeans:
        kmeans = KMeans(n_clusters=num_category, n_init = 1, n_jobs=8)
    else:
        kmeans = KMeans(n_clusters=num_category,n_jobs=8)

    embedding = np.array(embedding)
    kmeans.fit(embedding)
    y_kmeans_pred = kmeans.predict(embedding)
    nmi = normalized_mutual_info_score(label, y_kmeans_pred, average_method='arithmetic')
    return nmi


def eval_recall_K(embedding, label, K_list=None):
    if K_list is None:
        K_list = [1, 2, 4, 8]
    norm = np.sum(embedding * embedding, axis=1)
    right_num = 0

    recall_list = np.zeros(len(K_list))

    for i in range(embedding.shape[0]):
        dis = norm[i] + norm - 2 * np.squeeze(np.matmul(embedding[i], embedding.T))
        dis[i] = 1e10
        index = np.argsort(dis)
        list_index = 0
        for k in range(np.max(K_list)):
            if label[i] == label[index[k]]:
                recall_list[list_index] = recall_list[list_index] + 1
                break
            if k >= K_list[list_index] - 1:
                list_index = list_index + 1
    recall_list = recall_list / float(embedding.shape[0])
    for i in range(recall_list.shape[0]):
        if i == 0:
            continue
        recall_list[i] = recall_list[i] + recall_list[i - 1]
    return recall_list


convert_to_pil = transforms.ToPILImage()


def bestMap(L1, L2):
    # compute the accuracy
    if L1.__len__() != L2.__len__():
        print('size(L1) must == size(L2)')
    L1 = np.array(L1)
    L2 = np.array(L2)
    Label1 = np.unique(L1)
    nClass1 = Label1.__len__()
    Label2 = np.unique(L2)
    nClass2 = Label2.__len__()
    nClass = max(nClass1, nClass2)
    G = np.zeros((nClass, nClass))
    for i in range(nClass1):
        for j in range(nClass2):
            G[i][j] = np.nonzero((L1 == Label1[i]) * (L2 == Label2[j]))[0].__len__()

    c = linear_assignment_.linear_assignment(-G.T)[:, 1]
    newL2 = np.zeros(L2.__len__())
    for i in range(nClass2):
        for j in np.nonzero(L2 == Label2[i])[0]:
            if len(Label1) > c[i]:
                newL2[j] = Label1[c[i]]

    return accuracy_score(L1, newL2)


def l2norm(x):
    out = x.pow(2).sum(1, keepdim=True).pow(1/2)
    x = x.div(out)
    return x


class LossLogger:
    def __init__(self, name):
        self.name = name
        self.iter = 0

    def add_scalar(self, writer, value):
        writer.add_scalar(self.name, value, self.iter)
        self.iter += 1


def entropy(p):
    # compute entropy
    if (len(p.size())) == 2:
        return - torch.sum(p * torch.log(p + 1e-18)) / np.log(len(p)) / float(len(p))
    elif (len(p.size())) == 1:
        return - torch.sum(p * torch.log(p + 1e-18)) / np.log(len(p))
    else:
        raise NotImplementedError


def freeze_module(module):
    for param in module.parameters():
        param.requires_grad = False


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


inv_tra = transforms.Compose([
    # transforms.Lambda(lambda x: x * 255.0),
    transforms.Lambda(lambda x: x[[2, 1, 0], ...]),
    NormalizeInverse(mean=[122.7717, 115.9465, 102.9801], std=[1., 1., 1.]),
    transforms.Lambda(lambda x: x / 255),
])