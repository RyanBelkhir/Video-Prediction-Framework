from data.MNIST.MovingMNIST import MovingMNIST
from data.MNIST.stochastic_moving_mnist import StochasticMovingMNIST
from data.KTH.kth import KTHDataset
import os

import torch
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.optim as optim
import argparse
import yaml

def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace

# Argparse 
parser = argparse.ArgumentParser(description=globals()['__doc__'])
parser.add_argument('--config', type=str, required=True, help="Path to config file")
parser.add_argument('--checkpoint', type=bool, default=False, help="Checkpoint to resume training")
args = parser.parse_args()

# Load config file with models and training parameters
with open(args.config, "r") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)
    config = dict2namespace(config)


root = 'data/MNIST'

if config.data.dataset == "MovingMnist":
    train_set = MovingMNIST(data_root='.data/mnist', train=True, download=True)
    test_set = MovingMNIST(data_root='.data/mnist', train=False, download=True)

    train_loader = torch.utils.data.DataLoader(
                    dataset=train_set,
                    batch_size=config.training.batch_size,
                    shuffle=True)
    test_loader = torch.utils.data.DataLoader(
                    dataset=test_set,
                    batch_size=config.training.batch_size,
                    shuffle=False)

elif config.data.dataset == "StochasticMovingMNIST":
    train_set = StochasticMovingMNIST(data_root='.data/stochastic_mnist', train=True)
    test_set = StochasticMovingMNIST(data_root='.data/stochastic_mnist', train=False)

    train_loader = torch.utils.data.DataLoader(
                    dataset=train_set,
                    batch_size=config.training.batch_size,
                    shuffle=True)
    test_loader = torch.utils.data.DataLoader(
                    dataset=test_set,
                    batch_size=config.training.batch_size,
                    shuffle=False)

elif config.data.dataset == "KTH":
    num_frames = frames_per_sample=config.data.num_frames + config.data.num_frames_cond
    train_set = KTHDataset(data_dir=".data/kth/KTH64_h5", frames_per_sample=num_frames , train=True, with_target=False)
    test_set = KTHDataset(data_dir=".data/kth/KTH64_h5", frames_per_sample=num_frames, train=False, with_target=False)

    train_loader = torch.utils.data.DataLoader(
                    dataset=train_set,
                    batch_size=config.training.batch_size,
                    shuffle=True)
    test_loader = torch.utils.data.DataLoader(
                    dataset=test_set,
                    batch_size=config.training.batch_size,
                    shuffle=False)



print('==>>> total trainning batch number: {}'.format(len(train_loader)))
print('==>>> total testing batch number: {}'.format(len(test_loader)))

for seq in train_loader:
    print('--- Sample')
    print('Input:  ', seq.shape)
    break

from models.models import UNet
from models.ddpm import DDPM, EMA
from models.trainer import Trainer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = UNet(config).to(device)
ddpm = DDPM(config)
trainer = Trainer(model, ddpm)
trainer.train(train_loader)