import hydra
from hydra.utils import instantiate
import torch
import os
from omegaconf import OmegaConf
import wandb

from modeling.training import generate_samples, train_epoch

@hydra.main(config_path=".", config_name="main_config") # Добавили гидру
def main(clf):
    path = "samples" # Добавили папку для вывода
    if not os.path.exists(path):
        os.mkdir(path)
        print("Create dir")

    
    clf['device'] = "cuda" if torch.cuda.is_available() else "cpu"
    ddpm = instantiate(clf['DiffusionModel']).to(clf['device']) # Переписали под гидру

    train_transforms = instantiate(clf['Augmentations'])
    dataset = instantiate(clf['CIFAR10'], transform=train_transforms)

    dataloader = instantiate(clf['DataLoader'], dataset)
    optim = instantiate(clf['optimizer'], ddpm.parameters())
    
    wandb.login() # Добавили wandb
    wandb.init(project='hw_1', name=clf.name, config=OmegaConf.to_container(
        clf, resolve=True, throw_on_missing=True
    ))

    generate_samples(ddpm, clf['device'], f"{path}/0.png")
    noise = torch.randn((8, 3, 32, 32), device=clf['device'])
    wandb.log({"Generate Images": wandb.Image(noise)}, step=0)
    for i in range(clf['num_epochs']):
        images = train_epoch(ddpm, dataloader, optim, clf['device'], i, clf['logging_policy'])
        generate_samples(ddpm, clf['device'], f"{path}/{i+1:02d}.png")
        result = generate_samples(ddpm, clf['device'], f"{path}/{i+1:02d}.png", noise)
        wandb.log({"Inputs": wandb.Image(images)}, step=(i+1)*len(dataloader))
        wandb.log({"Generate Images": wandb.Image(result)}, step=(i+1)*len(dataloader))


if __name__ == "__main__":
    main()
