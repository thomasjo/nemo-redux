from argparse import ArgumentParser
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim

from ignite.contrib.handlers import WandBLogger
from ignite.engine import Engine, Events, create_supervised_evaluator, create_supervised_trainer
from ignite.metrics import Accuracy, Loss
from ignite.utils import setup_logger
from torch.utils.data import DataLoader, RandomSampler

from nemo.datasets import prepare_datasets
from nemo.models import initialize_classifier
from nemo.utils import ensure_reproducibility, timestamped_path


def main(args):
    ensure_reproducibility(seed=42)

    # Append timestamp to output directory.
    args.output_dir = timestamped_path(args.output_dir)
    args.output_dir.mkdir(parents=True)

    # Development mode overrides...
    log_interval = 1 if args.dev_mode else 10
    max_epochs = 2 if args.dev_mode else args.max_epochs
    epoch_length = 2 if args.dev_mode else None

    # TODO(thomasjo): Transition away from pre-partitioned datasets to on-demand partitioning.
    train_dataloader, val_dataloader, test_dataloader, num_classes = prepare_dataloaders(args.data_dir)

    model = initialize_classifier(num_classes)
    model.to(device=args.device)

    optimizer = optim.Adam(model.parameters(), lr=1e-5)
    criterion = nn.NLLLoss()

    trainer = create_supervised_trainer(
        model,
        optimizer,
        criterion,
        device=args.device,
        non_blocking=True,
    )

    def evaluator_transform(x, y, y_pred):
        return {"x": x, "y": y, "y_pred": y_pred}

    def metric_transform(output):
        return output["y_pred"], output["y"]

    metrics = metrics = {
        "loss": Loss(criterion, output_transform=metric_transform),
        "accuracy": Accuracy(),
    }

    train_evaluator = create_supervised_evaluator(model, metrics, device=args.device, non_blocking=True, output_transform=evaluator_transform)
    val_evaluator = create_supervised_evaluator(model, metrics, device=args.device, non_blocking=True, output_transform=evaluator_transform)

    @trainer.on(Events.EPOCH_COMPLETED)
    def compute_metrics(engine: Engine):
        train_evaluator.run(train_dataloader, max_epochs=max_epochs, epoch_length=epoch_length)
        val_evaluator.run(val_dataloader, max_epochs=max_epochs, epoch_length=epoch_length)

    # Configure basic logging.
    trainer.logger = setup_logger("trainer")
    train_evaluator.logger = setup_logger("train_evaluator")
    val_evaluator.logger = setup_logger("val_evaluator")

    @trainer.on(Events.ITERATION_COMPLETED(every=log_interval))
    def log_training_loss(engine: Engine):
        engine.logger.info("Epoch[{}] Iteration[{}/{}] Loss: {:.4f}".format(
            engine.state.epoch,
            engine.state.iteration,
            epoch_length,
            engine.state.output,
        ))

    # Configure W&B logging.
    wandb_logger = WandBLogger(dir=str(args.output_dir))
    wandb_logger.watch(model, criterion, log="all", log_freq=log_interval)

    wandb_logger.attach_opt_params_handler(
        trainer,
        event_name=Events.ITERATION_COMPLETED(every=log_interval),
        optimizer=optimizer,
    )

    wandb_logger.attach_output_handler(
        trainer,
        event_name=Events.ITERATION_COMPLETED(every=log_interval),
        tag="training",
        output_transform=lambda loss: {"batchloss": loss},
    )

    for tag, evaluator in [("training", train_evaluator), ("validation", val_evaluator)]:
        wandb_logger.attach_output_handler(
            evaluator,
            event_name=Events.EPOCH_COMPLETED,
            tag=tag,
            metric_names="all",
            global_step_transform=lambda *_: trainer.state.iteration,
        )

    # TODO(thomasjo): Save model weights; use ModelCheckpoint perhaps?
    # def score_function(engine):
    #     return engine.state.metrics["accuracy"]

    # Kick-off model training...
    trainer.run(train_dataloader, max_epochs=max_epochs, epoch_length=epoch_length)


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
