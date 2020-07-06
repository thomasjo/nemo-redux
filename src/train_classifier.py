from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim

from ignite.engine import Events
from ignite.engine import create_supervised_evaluator, create_supervised_trainer
from ignite.metrics import Accuracy, Loss
from torch.utils.data import DataLoader, RandomSampler
from torchvision.utils import save_image  # noqa
from tqdm.auto import tqdm

from nemo.datasets import prepare_datasets
from nemo.models import initialize_classifier
from nemo.utils import ensure_reproducibility


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
    criterion = nn.NLLLoss()

    max_epochs = 1 if args.dev else args.max_epochs
    epoch_length = 1 if args.dev else None

    metrics = {"accuracy": Accuracy(), "nll": Loss(criterion)}
    trainer = create_supervised_trainer(model, optimizer, criterion, device=device, non_blocking=True)
    evaluator = create_supervised_evaluator(model, metrics=metrics, device=device, non_blocking=True)

    desc = "ITERATION - loss: {:.2f}"
    pbar = tqdm(initial=0, leave=False, total=len(train_dataloader), desc=desc.format(0))

    @trainer.on(Events.ITERATION_COMPLETED(every=10))
    def log_training_loss(engine):
        pbar.desc = desc.format(engine.state.output)
        pbar.update(10)

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_training_results(engine):
        pbar.refresh()
        evaluator.run(train_dataloader, max_epochs=max_epochs, epoch_length=epoch_length)
        metrics = evaluator.state.metrics
        acc = metrics["accuracy"]
        nll = metrics["nll"]
        tqdm.write(f"Training Results - Epoch: {engine.state.epoch} Accuracy: {acc:.2f} Loss: {nll:.2f}")

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_validation_results(engine):
        pbar.refresh()
        evaluator.run(val_dataloader, max_epochs=max_epochs, epoch_length=epoch_length)
        metrics = evaluator.state.metrics
        acc = metrics["accuracy"]
        nll = metrics["nll"]
        tqdm.write(f"Validation Results - Epoch: {engine.state.epoch} Accuracy: {acc:.2f} Loss: {nll:.2f}")
        pbar.n = pbar.last_print_n = 0

    @trainer.on(Events.EPOCH_COMPLETED | Events.COMPLETED)
    def log_time(engine):
        event_name = trainer.last_event_name.name
        event_duration = trainer.state.times[event_name]
        tqdm.write(f"{event_name} took {event_duration} seconds")

    trainer.run(train_dataloader, max_epochs=max_epochs, epoch_length=epoch_length)
    evaluator.run(test_dataloader, max_epochs=max_epochs, epoch_length=epoch_length)

    pbar.close()


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
