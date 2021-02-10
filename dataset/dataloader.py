import glob
import os
from enum import Enum, auto

import hydra
import torch
import torchvision
from prefetch_generator import BackgroundGenerator
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
import torchvision.transforms as transforms
from torchvision.transforms.transforms import Resize
from dataset.autoaugment import CIFAR10Policy
from dataset.cutout import Cutout
# from randaugment import CIFAR10PolicyAll
from RandAugment import RandAugment

class DataloaderMode(Enum):
    train = auto()
    test = auto()
    inference = auto()


class DataLoader_(DataLoader):
    # ref: https://github.com/IgorSusmelj/pytorch-styleguide/issues/5#issuecomment-495090086
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())


def create_dataloader(cfg, mode, rank):
    if cfg.data.use_background_generator:
        data_loader = DataLoader_
    else:
        data_loader = DataLoader
    dataset = Dataset_(cfg, mode)
    train_use_shuffle = True
    sampler = None
    if cfg.dist.gpus > 0 and cfg.data.divide_dataset_per_gpu:
        sampler = DistributedSampler(dataset, cfg.dist.gpus, rank)
        train_use_shuffle = False
    if mode is DataloaderMode.train:
        return data_loader(
            dataset=dataset,
            batch_size=cfg.batch_size,
            shuffle=train_use_shuffle,
            sampler=sampler,
            num_workers=cfg.num_workers,
            pin_memory=True,
            drop_last=True,
        )
    elif mode is DataloaderMode.test:
        return data_loader(
            dataset=dataset,
            batch_size=cfg.batch_size,
            shuffle=False,
            sampler=sampler,
            num_workers=cfg.num_workers,
            pin_memory=True,
            drop_last=True,
        )
    else:
        raise ValueError(f"invalid dataloader mode {mode}")


class Dataset_(Dataset):
    def __init__(self, cfg, mode):
        self.cfg = cfg
        self.mode = mode
        self.transformer = {
            "train": transforms.Compose( 
                [
                    # RandAugment(3, 4),
                    transforms.RandomCrop(32, padding=4, fill=128), # fill parameter needs torchvision installed from source
                    transforms.RandomHorizontalFlip(), 

                    transforms.Resize(224),
                    # CIFAR10PolicyAll(),
                    transforms.ToTensor(), 
                    Cutout(n_holes=1, length=16),
                    transforms.Normalize(**(cfg.data[cfg.data.mode]))
                ]), # meanstd transformation

            "test" : transforms.Compose([
                    transforms.Resize(224),
                    transforms.ToTensor(),
                    transforms.Normalize(**(cfg.data[cfg.data.mode])),
                ])
        }
        # self.transformer["train"].insert(0, RandAugment(3, 5))
        if mode is DataloaderMode.train:
            # self.data_dir = self.cfg.data.train_dir
            # TODO: This is example code. You should change this part as you need
            self.dataset = torchvision.datasets.CIFAR100(
                root=hydra.utils.to_absolute_path("dataset/meta"),
                train=True,
                transform=self.transformer["train"],
                download=False,
            )
        elif mode is DataloaderMode.test:
            # self.data_dir = self.cfg.data.test_dir
            # TODO: This is example code. You should change this part as you need
            self.dataset = torchvision.datasets.CIFAR100(
                root=hydra.utils.to_absolute_path("dataset/meta"),
                train=False,
                transform=self.transformer["test"],
                download=False,
            )
        else:
            raise ValueError(f"invalid dataloader mode {mode}")
        # self.dataset_files = sorted(
        #     map(
        #         os.path.abspath,
        #         glob.glob(os.path.join(self.data_dir, self.cfg.data.file_format)),
        #     )
        # )
        # self.dataset = list()
        # for dataset_file in self.dataset_files:
        #     pass

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]
