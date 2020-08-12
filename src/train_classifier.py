from argparse import ArgumentParser
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import wandb

from ignite.engine import Events, create_supervised_evaluator, create_supervised_trainer
from ignite.metrics import Accuracy, Loss
from torch.utils.data import DataLoader, RandomSampler

from nemo.datasets import prepare_datasets
from nemo.models import initialize_classifier
from nemo.utils import ensure_reproducibility, timestamped_path


def main(args):
    ensure_reproducibility(seed=42)

    # Append timestamp to output directory.
    args.output_dir = timestamped_path(args.output_dir)
    args.output_dir.mkdir(parents=True)

    # TODO(thomasjo): Transition away from pre-partitioned datasets to on-demand partitioning.
    train_dataloader, val_dataloader, test_dataloader, num_classes = prepare_dataloaders(args.data_dir)

    wandb.init(dir=str(args.output_dir))

    model = initialize_classifier(num_classes)
    model.to(device=args.device)

    optimizer = optim.Adam(model.parameters(), lr=1e-5)
    criterion = nn.NLLLoss()

    metrics = metrics = {
        "accuracy": Accuracy(),
        "loss": Loss(criterion),
    }

    log_interval = 1 if args.dev_mode else 10
    max_epochs = 2 if args.dev_mode else args.max_epochs
    epoch_length = 2 if args.dev_mode else None

    wandb.watch(model, criterion, log_freq=log_interval)

    trainer = create_supervised_trainer(model, optimizer, criterion, device=args.device)
    evaluator = create_supervised_evaluator(model, metrics, device=args.device)

    @trainer.on(Events.ITERATION_COMPLETED(every=log_interval))
    def log_training_loss(engine):
        wandb.log({
            "train/loss": engine.state.output,
        })

        print("Epoch[{}] Iteration[{}/{}] Loss: {:.2f}".format(
            engine.state.epoch,
            engine.state.iteration,
            epoch_length,
            engine.state.output,
        ))

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_training_results(engine):
        evaluator.run(train_dataloader, max_epochs=max_epochs, epoch_length=epoch_length)

        metrics = evaluator.state.metrics
        avg_accuracy = metrics["accuracy"]
        avg_loss = metrics["loss"]

        wandb.log({
            "train/avg_accuracy": avg_accuracy,
            "train/avg_loss": avg_loss,
        })

        print("Training Results\tEpoch: {}\tAccuracy: {:.2f}\tLoss: {:.2f}".format(
            engine.state.epoch,
            avg_accuracy,
            avg_loss,
        ))

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_validation_results(engine):
        val_epochs = max_epochs if args.dev_mode else None
        val_epoch_length = epoch_length if args.dev_mode else None

        evaluator.run(val_dataloader, max_epochs=val_epochs, epoch_length=val_epoch_length)

        metrics = evaluator.state.metrics
        avg_accuracy = metrics["accuracy"]
        avg_loss = metrics["loss"]

        wandb.log({
            "val/avg_accuracy": avg_accuracy,
            "val/avg_loss": avg_loss,
        })

        print("Validation Results\tEpoch: {}\tAccuracy: {:.2f}\tLoss: {:.2f}".format(
            engine.state.epoch,
            avg_accuracy,
            avg_loss,
        ))

    @trainer.on(Events.EPOCH_COMPLETED | Events.COMPLETED)
    def log_time(engine):
        print("{} took {} seconds".format(trainer.last_event_name.name, trainer.state.times[trainer.last_event_name.name]))

    trainer.run(train_dataloader, max_epochs=max_epochs, epoch_length=epoch_length)

    # TODO(thomasjo): Save model weights; use ModelCheckpoint perhaps?


def prepare_dataloaders(data_dir, batch_size=32):
    train_dataset, val_dataset, test_dataset = prepare_datasets(args.data_dir)
    num_classes = len(train_dataset.classes)

    sampling_factor = 4  # Oversample to get more random augmentations
    num_training_samples = sampling_factor * len(train_dataset)
    train_sampler = RandomSampler(train_dataset, replacement=True, num_samples=num_training_samples)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler, num_workers=args.num_workers, pin_memory=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    return train_dataloader, val_dataloader, test_dataloader, num_classes


def parse_args():
    parser = ArgumentParser()

    parser.add_argument("--data-dir", type=Path, metavar="PATH", required=True, help="path to partitioned data directory")
    parser.add_argument("--output-dir", type=Path, metavar="PATH", required=True, help="path to output directory")
    parser.add_argument("--device", type=torch.device, metavar="NAME", default="cuda", help="device to use for model training")
    parser.add_argument("--num-workers", type=int, metavar="NUM", default=2, help="number of workers to use for data loaders")
    parser.add_argument("--max-epochs", type=int, metavar="NUM", default=25, help="maximum number of epochs to train")
    parser.add_argument("--dev-mode", action="store_true", help="run each model phase with only one batch")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
