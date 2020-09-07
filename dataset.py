import pickle
from os import path as osp
from torch.utils.data import Dataset
from torchvision.datasets.folder import default_loader
from augmentation import transform_train, transform_aug
from torchvision import transforms
import torch
import numpy as np


class CUB200_2011Dataset(Dataset):
    def __init__(self, root, train=True, transform=transform_train, nnIndex=None, augment=transform_train, target_transform=None):
        self.root = root
        self.dataset_dir = osp.join(self.root, self.__class__.__name__)
        self.transform = transform
        self.augment = augment
        self.target_transform = target_transform
        self.ae = False
        self.train = train
        self.loader = default_loader
        self.cls_to_idx = {}
        self.ps_label = []
        self._paths, self._targets = self.get_processed_data()
        self.nnIndex = nnIndex

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        tmp = 'train' if self.train is True else 'test'
        fmt_str += '    Split: {}\n'.format(tmp)
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str

    def __len__(self):
        return len(self.paths)

    def get_processed_data(self):
        self.gen_cls2idx()
        if self.train:
            data_file = 'train.txt'
        else:
            data_file = 'test.txt'
        paths = []
        targets = []
        with open(osp.join(self.dataset_dir, data_file)) as f:
            for line in f:
                filename = line.strip()
                target = self.cls_to_idx[osp.split(filename)[0]]
                filepath = osp.join(self.dataset_dir, 'images', filename)
                paths.append(filepath)
                targets.append(target)
        return paths, targets

    def gen_cls2idx(self):
        if osp.exists(osp.join(self.dataset_dir, 'cls2idx.dict')):
            self.cls_to_idx = pickle.load(open(osp.join(self.dataset_dir, 'cls2idx.dict'), 'rb'))
        else:

            with open(osp.join(self.dataset_dir, 'classes.txt'), 'r') as f:
                for line in f:
                    self.cls_to_idx[line.strip()] = len(self.cls_to_idx)
            pickle.dump(self.cls_to_idx, open(osp.join(self.dataset_dir, 'cls2idx.dict'), 'wb'))

    def __getitem__(self, idx):
        if self.ae:
            img = default_loader(self.paths[idx])
            img = self.augment(img)
            train_img = self.transform(img)
            target_img = self.target_transform(img)
            target = self.targets[idx]
            return target_img, train_img, target, idx
        if self.nnIndex is not None:
            img1 = default_loader(self.paths[idx])
            img2 = default_loader(self.paths[self.nnIndex[idx]])
            target = self.targets[idx]
            img1 = self.transform(img1)
            img2 = self.augment(img2)
            return img1, img2, target, idx
        else:
            img = default_loader(self.paths[idx])
            img = self.transform(img)
            target = self.targets[idx]
            return img, target, idx

    def update_ps_label(self, labels):
        self.ps_label += labels

    @property
    def paths(self):
        return self._paths

    @property
    def targets(self):
        return self._targets


def handle_cub200(root, train=True):
    data_dir = osp.join(root, 'CUB200_2011Dataset')
    cls2idx = pickle.load(open(osp.join(data_dir, 'cls2idx.dict'), 'rb'))
    if train:
        data_file = 'train.txt'
    else:
        data_file = 'test.txt'
    paths = []
    targets = []
    with open(osp.join(data_dir, data_file)) as f:
        for line in f:
            filename = line.strip()
            target = cls2idx[osp.split(filename)[0]]
            filepath = osp.join(data_dir, 'images', filename)
            paths.append(filepath)
            targets.append(target)
    return paths, targets


def handle_dataset_alpha(root, train=True, dataset_name='cub200'):
    print("Loading", dataset_name)
    data_dir = osp.join(root, dataset_name)
    if train:
        img_data = np.load(data_dir + '/{}_{}_256resized_img.npy'.format('training', dataset_name))
        img_label = np.load(data_dir + '/{}_{}_256resized_label.npy'.format('training', dataset_name))
    else:
        img_data = np.load(data_dir + '/{}_{}_256resized_img.npy'.format('validation', dataset_name))
        img_label = np.load(data_dir + '/{}_{}_256resized_label.npy'.format('validation', dataset_name))
    return img_data, img_label


class MetricLearningDataset(Dataset):
    def __init__(self, root, train=True, dataset_name='cub200', transform=transform_train, augmentation=transform_train, nn_index=None):
        super(MetricLearningDataset, self).__init__()
        self.data, self.targets = handle_dataset_alpha(root, train, dataset_name)
        # nearest neighbor index
        self.nnIndex = nn_index
        if not self.nnIndex:
            self.nnIndex = np.arange(len(self.data))
        self.transform = transform
        self.augment = augmentation

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img1 = self.data[idx]
        img2 = self.data[self.nnIndex[idx]]
        target = self.targets[idx]
        img1 = self.transform(img1)
        img2 = self.augment(img2)
        return img1, img2, target


class AutoencoderDataset(Dataset):
    def __init__(self, root, train=True, dataset_name='cub200', input_sie=224, target_size=224):
        self.data, self.targets = handle_dataset_alpha(root, train, dataset_name)

        normalize = transforms.Compose([
            transforms.Lambda(lambda x: x * 255.0),
            transforms.Normalize(mean=[122.7717, 115.9465, 102.9801], std=[1, 1, 1]),
            transforms.Lambda(lambda x: x[[2, 1, 0], ...]),
            # transforms.Lambda(lambda x: x / 255.0),

        ])
        if train:
            self.augment = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((input_sie, input_sie)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(30),
            ])
        else:
            self.augment = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((input_sie, input_sie)),
            ])

        self.transform_target = transforms.Compose([
            transforms.Resize((target_size, target_size)),
            transforms.ToTensor(),
            normalize
        ])

        self.transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x + torch.zeros(x.size()).normal_(0, 0.09)),
            normalize
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = self.data[idx]
        img = self.augment(img)
        train_img = self.transform_train(img)
        target_img = self.transform_target(img)
        return train_img, target_img


class MLDataInstance(Dataset):
    """Metric Learning Dataset.
    """

    def __init__(self, src_dir, dataset_name, train=True, transform=None, target_transform=None, nnIndex=None,
                 augmenter=None):

        data_dir = src_dir + '/' + dataset_name + '/'
        if train:
            img_data = np.load(data_dir + '{}_{}_256resized_img.npy'.format('training', dataset_name))
            img_label = np.load(data_dir + '{}_{}_256resized_label.npy'.format('training', dataset_name))
        else:
            img_data = np.load(data_dir + '{}_{}_256resized_img.npy'.format('validation', dataset_name))
            img_label = np.load(data_dir + '{}_{}_256resized_label.npy'.format('validation', dataset_name))

        self.img_data = img_data
        self.targets = img_label
        self.transform = transform
        self.target_transform = target_transform
        self.nnIndex = nnIndex
        self.augmenter = augmenter

    def __getitem__(self, index):

        if self.nnIndex is not None:

            img1, img2, target = self.img_data[index], self.img_data[self.nnIndex[index]], self.targets[index]

            img1 = self.transform(img1)
            if self.augmenter:
                img2 = self.augmenter(img2)
            else:
                img2 = self.transform(img2)
            if self.target_transform is not None:
                target = self.target_transform(target)
            return img1, img2, target

        else:
            img, target = self.img_data[index], self.targets[index]
            img = self.transform(img)
            if self.target_transform is not None:
                target = self.target_transform(target)
            return img, img, target

    def __len__(self):
        return len(self.img_data)
