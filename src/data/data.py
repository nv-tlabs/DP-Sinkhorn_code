# ---------------------------------------------------------------
# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# for DP-Sinkhorn. To view a copy of this license, see the LICENSE file.
# ---------------------------------------------------------------
import torch
from torchvision.datasets import MNIST, FashionMNIST, CIFAR10, SVHN
from .CelebA import MyCelebA
import torchvision.transforms as transforms

import numpy as np
import os


def add_data_args(parser):
    parser.add_argument("--dataset", type=str,
                        choices=['mnist', 'fashion_mnist', 'cifar', 'svhn', 'svhn_28', 'celeb_32', 'celeb_32_2',
                                 'stackedmnist', 'mnist_bce', 'fashion_mnist_bce'], default='mnist')
    parser.add_argument("--datadir", type = str, default='datasets')
    return parser


datasets = {
    'mnist': {
        'transform': transforms.Compose([transforms.ToTensor(),
                                         transforms.Normalize((0.5,), (0.5,))]),
        'num_examples': 60000,
        'fetch_fn': MNIST,
        'metadata': {
            'img_dim': (1, 28, 28),
            'label_dim': 10,
            'display_labels': None,
            'cmap': 'gray'
        }
    },
    'fashion_mnist': {
        'transform': transforms.Compose([transforms.ToTensor(),
                                         transforms.Normalize((0.5,), (0.5,))]),
        'num_examples': 60000,
        'fetch_fn': FashionMNIST,
        'metadata': {
            'img_dim': (1, 28, 28),
            'label_dim': 10,
            'display_labels': ['t-shirt/top', 'trouser', 'pullover', 'dress', 'coat', 'sandal', 'shirt', 'sneaker',
                               'bag', 'ankleboot'],
            'cmap': 'gray'
        }
    },
    'mnist_bce': {
        'transform': transforms.Compose([transforms.ToTensor(),]),
        'num_examples': 60000,
        'fetch_fn': MNIST,
        'metadata': {
            'img_dim': (1, 28, 28),
            'label_dim': 10,
            'display_labels': None,
            'cmap': 'gray'
        }
    },
    'fashion_mnist_bce': {
        'transform': transforms.Compose([transforms.ToTensor(),]),
        'num_examples': 60000,
        'fetch_fn': FashionMNIST,
        'metadata': {
            'img_dim': (1, 28, 28),
            'label_dim': 10,
            'display_labels': ['t-shirt/top', 'trouser', 'pullover', 'dress', 'coat', 'sandal', 'shirt', 'sneaker',
                               'bag', 'ankleboot'],
            'cmap': 'gray'
        }
    },
    'cifar': {
        'transform': transforms.Compose([transforms.ToTensor(),
                                         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
        'num_examples': 50000,
        'fetch_fn': CIFAR10,
        'metadata': {
            'img_dim': (3, 32, 32),
            'label_dim': 10,
            'display_labels': None,
            'cmap': None
        }
    },
    'svhn': {
        'transform': transforms.Compose([transforms.ToTensor(),
                                         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
        'num_examples': 73257,
        'fetch_fn': SVHN,
        'metadata': {
            'img_dim': (3, 32, 32),
            'label_dim': 10,
            'display_labels': None,
            'cmap': None
        }
    },
    'svhn_28': {
        'transform': transforms.Compose([transforms.Resize((28, 28)),
                                         transforms.ToTensor(),
                                         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
        'num_examples': 73257,
        'fetch_fn': SVHN,
        'metadata': {
            'img_dim': (3, 28, 28),
            'label_dim': 10,
            'display_labels': None,
            'cmap': None
        }
    },

    'celeb_32_2': {
        'transform': transforms.Compose([transforms.Resize((32, 32)),
                                         transforms.ToTensor(),
                                         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
        'num_examples': 60000,
        'fetch_fn': MyCelebA,
        'metadata': {
            'img_dim': (3, 32, 32),
            'label_dim': 2,
            'display_labels': None,
            'cmap': None
        }
    },
}


def fetch_data(dataset_name, datadir, training=True, download=True, as_array=False, num_examples=0):
    """
    Produce either a dataset or array. If return as dataset, the iterable returns image-target tuples with preprocessing
    as specified in datasets. If return as array, return a pair of arrays with images preprocessed to range 0-1 float
    instead. Tries to do as few converions as possible.
    :param dataset_name: select one of the datasets available
    :param datadir: root directory containing directories for raw data
    :param training: returns training split if true, else return validation split
    :param download: allow redownload
    :param as_array: return as array
    :param num_examples: number of examples to take, should only be set when as_array is true, set to <=0 for everything
    :return:
    """
    dataset = datasets[dataset_name]

    path = os.path.join(datadir, dataset_name.split('_')[0])
    if dataset_name in ['svhn', 'shvn_28', 'celeb_32', 'celeb_32_2']:
        s = 'train' if training else 'valid'
        train_data = dataset['fetch_fn'](path, split=s, transform=dataset['transform'],
                                         download=not 'celeb' in dataset_name)
    else:
        train_data = dataset['fetch_fn'](path, train=training, transform=dataset['transform'],
                                         download=download)

    if as_array:
        if dataset_name in ['mnist', 'cifar', 'svhn', 'fashion_mnist', 'mnist_bce', 'fashion_mnist_bce']:
            x = train_data.data
            x = np.array(x, dtype=np.float32) / 255.0
            y = np.array(train_data.targets, dtype=int)
            if num_examples > 0:
                inds = np.arange(len(x))[:num_examples]
                x = x[inds]
                y = y[inds]

        elif dataset_name in ['celeb_32_2', 'svhn_28']:
            xs = []
            ys = []
            if num_examples > 0:
                inds = np.arange(len(train_data))[:num_examples]
                for i in inds:
                    x, y = train_data[i]
                    xs.append(x.numpy()*0.5 + 0.5)
                    ys.append(y.numpy())
            else:
                for x, y in train_data:
                    xs.append(x.numpy()*0.5 + 0.5)
                    ys.append(y.numpy())
            x = np.stack(xs)
            y = np.stack(ys)
        else:
            raise ValueError("Dataset %s not recognized" % dataset_name)

        print('| loaded {} dataset'.format(dataset_name))
        print('| training examples: {}'.format(len(x)))
        if len(x.shape) == 3:
            x = np.expand_dims(x, 1)
        return (x,y), dataset['metadata']
    else:
        if num_examples > 0:
            import warnings
            warnings.warn("num examples is not zero but not returning as array")
        print('| loaded {} dataset'.format(dataset_name))
        print('| training examples: {}'.format(len(train_data)))
        return train_data, dataset['metadata']


def get_balanced_class_labels(batch_size, num_classes):
    gen_label = torch.cat(
        (batch_size // num_classes) * [torch.arange(0, num_classes)] + \
        [torch.randperm(num_classes)[batch_size % num_classes]]
    ).long()

    return gen_label


class IndexedDataset:
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        batch = self.dataset.__getitem__(idx)
        return batch + (idx,)


class DualBatchSampler:
    def __init__(self, dataset, batch_size, num_iterations):
        self.length = len(dataset)
        self.batch_size = batch_size
        self.num_iterations = num_iterations

    def __iter__(self):
        for _ in range(self.num_iterations):
            indices = np.where(torch.rand(self.length) < (self.batch_size / self.length))[0]
            if indices.size > 0:
                indices2 = np.where(torch.rand(self.length) < (self.batch_size / self.length))[0]
                if indices2.size > 0:
                    yield np.concatenate([indices, indices2])

    def __len__(self):
        return self.num_iterations

class IIDBatchSampler:
    def __init__(self, dataset, batch_size, num_iterations):
        self.length = len(dataset)
        self.batch_size = batch_size
        self.num_iterations = num_iterations

    def __iter__(self):
        for _ in range(self.num_iterations):
            indices = np.where(torch.rand(self.length) < (self.batch_size / self.length))[0]
            if indices.size > 0:
                yield indices

    def __len__(self):
        return self.num_iterations


class FixedSizeBatchSampler:
    def __init__(self, dataset, batch_size, num_iterations):
        self.length = len(dataset)
        self.batch_size = batch_size
        self.num_batches = self.length // self.batch_size
        self.num_iterations = num_iterations

    def __iter__(self):
        for _ in range(self.num_iterations):
            batch_idx = torch.randint(0, self.num_batches, (1,)).item()
            indices = torch.arange(batch_idx * self.batch_size, (batch_idx + 1) * self.batch_size)
            yield indices

    def __len__(self):
        return self.num_iterations
