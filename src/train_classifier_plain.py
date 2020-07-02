from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader, RandomSampler
from torchvision.utils import save_image  # noqa
from tqdm.auto import trange, tqdm
from tqdm.contrib import tenumerate

from nemo.datasets import prepare_datasets
from nemo.models import initialize_classifier
from nemo.utils import ensure_reproducibility


def predict_one_batch(batch, model, criterion, optimizer=None):
    image, label = batch

    output = model(image)
    loss = criterion(output, label)

    if optimizer:
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return output, loss


def copy_to_device(tensor, device):
    return [t.to(device=device, non_blocking=True) for t in tensor]


def main(args):
    ensure_reproducibility(seed=42)

    device = torch.device(args.device)

    # TODO(thomasjo): Transition away from pre-partitioned datasets to on-demand partitioning.
    train_dataset, val_dataset, test_dataset = prepare_datasets(args.data_dir)
    num_classes = len(train_dataset.classes)

    train_sampler = RandomSampler(train_dataset, replacement=True, num_samples=4 * len(train_dataset))
    train_dataloader = DataLoader(train_dataset, batch_size=32, sampler=train_sampler, num_workers=args.num_workers, pin_memory=True)

    val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    model = initialize_classifier(num_classes)
    model = model.to(device=device)

    optimizer = optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss()

    max_batch_idx = 0 if args.dev else -1

    for epoch_idx in range(args.max_epochs):
        print(f"Epoch {epoch_idx + 1}")

        model.train()
        for batch_idx, batch in enumerate(train_dataloader):
            print(f"==> Training: Batch {batch_idx + 1}")
            batch = copy_to_device(batch, device)
            output, loss = predict_one_batch(batch, model, criterion, optimizer)

            # Short-circuit training.
            if batch_idx == max_batch_idx:
                break

        with torch.no_grad():
            model.eval()
            for batch_idx, batch in enumerate(val_dataloader):
                print(f"==> Validation: Batch {batch_idx + 1}")
                batch = copy_to_device(batch, device)
                output, loss = predict_one_batch(batch, model, criterion)

                # Short-circuit validation.
                if batch_idx == max_batch_idx:
                    break

        # Short-circuit epoch loop.
        if epoch_idx == max_batch_idx:
            break


def parse_args():
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument("--data-dir", type=Path, required=True, help="path to partitioned data directory")
    parser.add_argument("--max-epochs", type=int, default=25, help="maximum number of epochs to train")

    parser.add_argument("--dev", action="store_true", help="run each training phase with only one batch")
    parser.add_argument("--device", type=str, default="cuda", help="device to use for model training")
    parser.add_argument("--num-workers", type=int, default=2, help="number of workers to use for data loaders")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
