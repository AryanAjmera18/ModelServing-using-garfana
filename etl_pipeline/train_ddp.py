# File: train_ddp.py

import os
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from src.dataset import EyeDataset
from src.model import build_model
import mlflow.pytorch
import argparse
from tqdm import tqdm


def setup(rank, world_size):
    dist.init_process_group("nccl" if torch.cuda.is_available() else "gloo",
                            rank=rank, world_size=world_size)
    torch.manual_seed(0)

def cleanup():
    dist.destroy_process_group()

def train(rank, world_size, args):
    setup(rank, world_size)
    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")

    dataset = EyeDataset("data/processed")
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    loader = DataLoader(dataset, batch_size=args.batch_size, sampler=sampler)

    model = build_model(num_classes=10).to(device)
    model = DDP(model, device_ids=[rank] if torch.cuda.is_available() else None)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()

    if rank == 0:
        mlflow.set_experiment("eye-disease-ddp")
        mlflow.start_run()

    for epoch in range(args.epochs):
        model.train()
        sampler.set_epoch(epoch)
        running_loss = 0.0
        loop = tqdm(loader, desc=f"Rank {rank} Epoch {epoch+1}", disable=(rank != 0))

        for images, labels in loop:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if rank == 0:
                loop.set_postfix(loss=loss.item())

        if rank == 0:
            mlflow.log_metric("loss", running_loss / len(loader), step=epoch)

    if rank == 0:
        torch.save(model.module.state_dict(), "resnet50_ddp.pth")
        mlflow.pytorch.log_model(model.module, "resnet50_ddp")
        mlflow.end_run()

    cleanup()


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--gpus", type=int, default=1, help="number of GPUs")
    args = parser.parse_args()

    mp.spawn(train, args=(args.gpus, args), nprocs=args.gpus, join=True)

if __name__ == "__main__":
    main()
