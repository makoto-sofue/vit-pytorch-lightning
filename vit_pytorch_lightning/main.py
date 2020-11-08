import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms, datasets


import pytorch_lightning as pl
from einops import rearrange, repeat

from vit_pytorch_lightning import ViT

def CIFAR10dataset(batch_size=256, num_workers=4):
    transform = transforms.Compose([
      transforms.ToTensor()
    ])

    # データセットの取得
    train_val = datasets.CIFAR10('./', train=True, download=True, transform=transform)
    test = datasets.CIFAR10('./', train=False, download=True, transform=transform)

    # train と val に分割
    torch.manual_seed(0)
    n_train, n_val = 40000, 10000
    train, val = torch.utils.data.random_split(train_val, [n_train, n_val])

    # Data Loader
    train_loader = torch.utils.data.DataLoader(train, batch_size, shuffle=True, drop_last=True, num_workers=num_workers)
    val_loader = torch.utils.data.DataLoader(val, batch_size, num_workers=num_workers)
    test_loader = torch.utils.data.DataLoader(test, batch_size, num_workers=num_workers)

    return train_loader, val_loader, test_loader

def main(train_loader, val_loader, test_loader):
    pl.seed_everything(0)
    vit = ViT(dim=16, depth=12, heads=8, image_size=32, patch_size=4, num_classes=10, channels=3, mlp_dim=64)
    trainer = pl.Trainer(max_epochs=1)
    trainer.fit(vit, train_loader, val_loader)

    results = trainer.test(test_dataloaders=test_loader)
    print(results)

if __name__ == '__main__':
    train, val, test = CIFAR10dataset()
    main(train, val, test)
