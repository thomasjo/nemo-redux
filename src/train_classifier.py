from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim

from ignite.contrib.engines.common import add_early_stopping_by_val_score, save_best_model_by_val_score, setup_common_training_handlers, setup_wandb_logging
from ignite.engine import Events, create_supervised_evaluator, create_supervised_trainer
from ignite.metrics import Accuracy, Loss
from torch.utils.data import DataLoader, RandomSampler
from torchvision.utils import save_image  # noqa

from nemo.datasets import prepare_datasets
from nemo.models import initialize_classifier
from nemo.utils import ensure_reproducibility


def get_dataflow(args):
    # TODO(thomasjo): Transition away from pre-partitioned datasets to on-demand partitioning.
    train_dataset, val_dataset, test_dataset = prepare_datasets(args.data_dir)
    num_classes = len(train_dataset.classes)

    num_training_samples = 4 * len(train_dataset)  # Oversample to get more random augmentations
    train_sampler = RandomSampler(train_dataset, replacement=True, num_samples=num_training_samples)

    train_dataloader = DataLoader(train_dataset, batch_size=32, sampler=train_sampler, num_workers=args.num_workers, pin_memory=True)
    val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    return train_dataloader, val_dataloader, test_dataloader, num_classes


def initialize(num_classes, device):
    model = initialize_classifier(num_classes)
    model = model.to(device=device)

    optimizer = optim.Adam(model.parameters())
    criterion = nn.NLLLoss()

    return model, optimizer, criterion


def main(args):
    ensure_reproducibility(seed=42)

    device = torch.device(args.device)
    max_epochs = 1 if args.dev else args.max_epochs
    epoch_length = 1 if args.dev else None

    train_dataloader, val_dataloader, test_dataloader, num_classes = get_dataflow(args)
    model, optimizer, criterion = initialize(num_classes, device)
    metrics = {"accuracy": Accuracy(), "loss": Loss(criterion)}

    trainer = create_supervised_trainer(model, optimizer, criterion, device=device, non_blocking=True)
    evaluator = create_supervised_evaluator(model, metrics=metrics, device=device, non_blocking=True)

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_validation_results(engine):
        evaluator.run(train_dataloader, max_epochs=max_epochs, epoch_length=epoch_length)
        evaluator.run(val_dataloader, max_epochs=max_epochs, epoch_length=epoch_length)

    setup_common_training_handlers(trainer, log_every_iters=1)
    add_early_stopping_by_val_score(3, evaluator, trainer, metric_name="loss")
    save_best_model_by_val_score(args.output_dir, evaluator, model, metric_name="loss", trainer=trainer)
    setup_wandb_logging(trainer, optimizer, evaluator, log_every_iters=1)

    trainer.run(train_dataloader, max_epochs=max_epochs, epoch_length=epoch_length)
    evaluator.run(test_dataloader, max_epochs=max_epochs, epoch_length=epoch_length)


def parse_args():
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument("--data-dir", type=Path, required=True, help="path to partitioned data directory")
    parser.add_argument("--output-dir", type=Path, required=True, help="path to output directory")
    parser.add_argument("--max-epochs", type=int, default=25, help="maximum number of epochs to train")

    parser.add_argument("--dev", action="store_true", help="run each training phase with only one batch")
    parser.add_argument("--device", type=str, default="cuda", help="device to use for model training")
    parser.add_argument("--num-workers", type=int, default=2, help="number of workers to use for data loaders")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
