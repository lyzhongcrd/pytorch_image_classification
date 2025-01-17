import torch

from .defaults import get_default_config


def update_config(config):
    if config.dataset.name in ['CIFAR10', 'CIFAR100']:
        dataset_dir = f'/data0/datasets/{config.dataset.name}'
        config.dataset.dataset_dir = dataset_dir
        config.dataset.image_size = 32
        config.dataset.n_channels = 3
        config.dataset.n_classes = int(config.dataset.name[5:])
    elif config.dataset.name in ['MNIST', 'FashionMNIST', 'KMNIST']:
        dataset_dir = '/data0/datasets/'
        config.dataset.dataset_dir = dataset_dir
        config.dataset.image_size = 28
        config.dataset.n_channels = 1
        config.dataset.n_classes = 10
    elif config.dataset.name == 'GTSRB':
        dataset_dir = '/data0/datasets'
        config.dataset.dataset_dir = dataset_dir
        config.dataset.image_size = 48
        config.dataset.n_channels = 3
        config.dataset.n_classes = 43
    if not torch.cuda.is_available():
        config.device = 'cpu'

    return config
