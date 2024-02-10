import os

import pytest
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor, Normalize
from torchvision.datasets import CIFAR10

from modeling.diffusion import DiffusionModel
from modeling.training import train_step, train_epoch, generate_samples
from modeling.unet import UnetModel


@pytest.fixture
def train_dataset():
    transforms = Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    dataset = CIFAR10(
        "./data",
        train=True,
        download=True,
        transform=transforms,
    )
    return dataset


@pytest.mark.parametrize(["device"], [["cpu"], ["cuda"]])
def test_train_on_one_batch(device, train_dataset):
    # note: you should not need to increase the threshold or change the hyperparameters
    ddpm = DiffusionModel(
        eps_model=UnetModel(3, 3, hidden_size=32),
        betas=(1e-4, 0.02),
        num_timesteps=1000,
    )
    ddpm = ddpm.to(device) # fix

    optim = torch.optim.Adam(ddpm.parameters(), lr=5e-4)
    dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True)

    x, _ = next(iter(dataloader))
    loss = None
    for i in range(50):
        loss = train_step(ddpm, x, optim, device)
    assert loss < 0.5

@pytest.mark.parametrize(["device"], [["cpu"], ["cuda"]])
def test_training(device, train_dataset):
    # fix нагло копируем с test_train_on_one_batch
    # note: implement and test a complete training procedure (including sampling)
    
    path = 'test_dir'
    if not os.path.exists(path):
        os.mkdir(path)
    
    ddpm = DiffusionModel(
        eps_model=UnetModel(3, 3, hidden_size=32),
        betas=(1e-4, 0.02),
        num_timesteps=2,
    )
    ddpm = ddpm.to(device)

    optim = torch.optim.Adam(ddpm.parameters(), lr=5e-4)
    dataloader = DataLoader(torch.utils.data.Subset(train_dataset, list(range(3))), batch_size=4, shuffle=True)

    
    for i in range(5000): # Смотрим на последние картинки
        train_epoch(ddpm, dataloader, optim, device)
    generate_samples(ddpm, device, f"{path}/{device}_{i}.png")
