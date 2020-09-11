import os  # noqa

from argparse import ArgumentParser, Namespace
from math import ceil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from ignite.contrib.handlers import WandBLogger
from ignite.engine import Engine, EventEnum, Events, create_supervised_evaluator, create_supervised_trainer
from ignite.handlers import Checkpoint, DiskSaver
from ignite.metrics import Accuracy, Loss, RunningAverage
from ignite.utils import setup_logger
from matplotlib.backends.backend_agg import FigureCanvasAgg
from torch.utils.data import DataLoader

from nemo.datasets import prepare_datasets
from nemo.models import initialize_classifier
from nemo.utils import ensure_reproducibility, random_state_protection, timestamp_path


class CustomEvents(EventEnum):
    EXAMPLE_DATA_READY = "example_data_ready"
    EXAMPLE_PREDICTIONS_READY = "example_predictions_ready"


def main(args):
    ensure_reproducibility(seed=42)

    # Append timestamp to output directory.
    args.output_dir = timestamp_path(args.output_dir)
    args.output_dir.mkdir(parents=True)

    # Development mode overrides.
    args.log_interval = 1 if args.dev_mode else 10
    args.max_epochs = 2 if args.dev_mode else args.max_epochs
    args.epoch_length = 2 if args.dev_mode else None

    # TODO(thomasjo): Transition away from pre-partitioned datasets to on-demand partitioning.
    train_dataloader, val_dataloader, test_dataloader = prepare_dataloaders(args.data_dir, num_workers=args.num_workers)
    num_classes = len(train_dataloader.dataset.classes)

    model = initialize_classifier(num_classes)
    model.to(device=args.device)

    optimizer = optim.Adam(model.parameters(), lr=1e-5)
    criterion = nn.NLLLoss()

    metrics = metrics = {
        "loss": Loss(criterion, output_transform=metric_transform),
        "accuracy": Accuracy(output_transform=metric_transform),
    }

    trainer = create_trainer(model, optimizer, criterion, metrics, args)
    evaluator = create_evaluator(model, metrics, args)
    test_evaluator = create_evaluator(model, metrics, args, name="test_evaluator")

    # Register custom events.
    trainer.register_events(*CustomEvents)

    @trainer.on(Events.EPOCH_COMPLETED)
    def compute_validation_metrics(engine: Engine):
        # Development mode overrides.
        max_epochs = args.max_epochs if args.dev_mode else None
        epoch_length = args.epoch_length if args.dev_mode else None
        evaluator.run(val_dataloader, max_epochs=max_epochs, epoch_length=epoch_length)

    @trainer.on(Events.COMPLETED)
    def compute_test_metrics(engine: Engine):
        test_evaluator.run(test_dataloader, max_epochs=args.max_epochs, epoch_length=args.epoch_length)

    @trainer.on(Events.ITERATION_COMPLETED(every=args.log_interval))
    def log_training_metrics(engine: Engine):
        engine.logger.info("Epoch[{}] Iteration[{}/{}] Loss: {:.4f} Accuracy: {:.4f}".format(
            engine.state.epoch,
            engine.state.iteration,
            engine.state.max_epochs * engine.state.epoch_length,
            engine.state.metrics["loss"],
            engine.state.metrics["accuracy"],
        ))

    configure_checkpoint_saving(trainer, evaluator, model, optimizer, args)
    configure_example_predictions(trainer, train_dataloader, val_dataloader, model, args)
    configure_wandb_logging(trainer, evaluator, test_evaluator, model, criterion, optimizer, args)

    # Kick off the whole model training shebang...
    trainer.run(train_dataloader, max_epochs=args.max_epochs, epoch_length=args.epoch_length)


def prepare_dataloaders(data_dir, batch_size=32, num_workers=None):
    train_dataset, val_dataset, test_dataset = prepare_datasets(data_dir)

    # Embed pre-calculated training dataset moments into dataset object.
    train_dataset.moments = {"mean": [0.232, 0.244, 0.269], "std": [0.181, 0.182, 0.190]}

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    return train_dataloader, val_dataloader, test_dataloader


def create_trainer(model, optimizer, criterion, metrics, args):
    trainer = create_supervised_trainer(
        model,
        optimizer,
        criterion,
        device=args.device,
        non_blocking=True,
        output_transform=trainer_transform,
    )

    # Configure default engine output logging.
    trainer.logger = setup_logger("trainer")

    # Compute running averages of metrics during training.
    for name, metric in metrics.items():
        RunningAverage(metric).attach(trainer, name)

    return trainer


def create_evaluator(model, metrics, args, name="evaluator"):
    evaluator = create_supervised_evaluator(
        model,
        metrics,
        device=args.device,
        non_blocking=True,
        output_transform=evaluator_transform,
    )

    # Configure default engine output logging.
    evaluator.logger = setup_logger(name)

    return evaluator


def configure_checkpoint_saving(trainer, evaluator, model, optimizer, args):
    to_save = {"model": model, "optimizer": optimizer}
    save_handler = DiskSaver(str(args.output_dir), create_dir=False, require_empty=False)

    # Configure epoch checkpoints.
    interval = 1 if args.dev_mode else min(5, args.max_epochs)
    checkpoint = Checkpoint(to_save, save_handler, n_saved=None, global_step_transform=lambda *_: trainer.state.epoch)
    trainer.add_event_handler(Events.EPOCH_COMPLETED(every=interval), checkpoint, evaluator)

    # Configure "best score" checkpoints.
    metric_name = "accuracy"
    best_checkpoint = Checkpoint(to_save, save_handler, score_name=metric_name, score_function=lambda engine: engine.state.metrics[metric_name], filename_prefix="best")
    trainer.add_event_handler(Events.EPOCH_COMPLETED, best_checkpoint, evaluator)


def configure_example_predictions(trainer: Engine, train_dataloader, val_dataloader, model, args):
    example_batch_size = 64
    with random_state_protection():
        train_examples = grab_shuffled_data(train_dataloader, example_batch_size, args)
        val_examples = grab_shuffled_data(val_dataloader, example_batch_size, args)

    @trainer.on(Events.EPOCH_STARTED)
    def store_examples(engine: Engine):
        engine.state.examples = {"training": train_examples, "validation": val_examples}
        engine.logger.info("Example data ready")
        engine.fire_event(CustomEvents.EXAMPLE_DATA_READY)

    @trainer.on(Events.EPOCH_COMPLETED)
    def predict_on_examples(engine: Engine):
        model.eval()
        for tag, batch in engine.state.examples.items():
            x, y = batch
            with torch.no_grad():
                y_pred = model(x.to(args.device, non_blocking=True))
                y_pred = y_pred.detach().cpu()
            engine.state.examples[tag] = (x, y, y_pred)
        engine.logger.info("Example predictions ready")
        engine.fire_event(CustomEvents.EXAMPLE_PREDICTIONS_READY)


def configure_wandb_logging(trainer, evaluator, test_evaluator, model, criterion, optimizer, args):
    if args.dev_mode:
        os.environ["WANDB_MODE"] = "dryrun"

    wandb_logger = WandBLogger(dir=str(args.output_dir))
    wandb_logger.watch(model, criterion, log="all", log_freq=args.log_interval)

    # Log training-specific metrics.
    wandb_logger.attach_output_handler(
        trainer,
        event_name=Events.ITERATION_COMPLETED(every=args.log_interval),
        tag="training",
        output_transform=lambda output: {"batchloss": output["loss"]},
        global_step_transform=lambda *_: trainer.state.iteration,
    )

    # Configure basic metric logging.
    for tag, engine in [("training", trainer), ("validation", evaluator), ("test", test_evaluator)]:
        wandb_logger.attach_output_handler(
            engine,
            event_name=Events.EPOCH_COMPLETED,
            tag=tag,
            metric_names="all",
            global_step_transform=lambda *_: trainer.state.iteration,
        )

    # Track the epoch associated with the current training iteration.
    @trainer.on(Events.ITERATION_STARTED(every=args.log_interval))
    def log_epoch(engine: Engine):
        wandb_logger.log({"epoch": engine.state.epoch}, step=engine.state.iteration, commit=False)

    @trainer.on(CustomEvents.EXAMPLE_PREDICTIONS_READY)
    def log_example_predictions(engine: Engine):
        for tag, (x, y, y_pred) in engine.state.examples.items():
            x, y, y_pred = x.numpy(), y.numpy(), y_pred.numpy()

            # Convert log scale (torch.log_softmax) predictions.
            y_pred = np.exp(y_pred)

            # Prepare images for plotting.
            moments = engine.state.dataloader.dataset.moments
            x = x.transpose(0, 2, 3, 1)  # NCHW -> NHWC
            x = x * moments["std"] + moments["mean"]  # Denormalize using dataset moments
            x = x.clip(0, 1)

            # Plot grid of predictions for "example" batch.
            idx_to_class = {v: k for k, v in engine.state.dataloader.dataset.class_to_idx.items()}
            image = prediction_grid(x, y, y_pred, idx_to_class)

            # Save the prediction grid both to file system and W&B.
            wandb_logger.log({f"{tag}/examples": wandb_logger.Image(image)}, step=engine.state.iteration)


def prediction_grid(x: np.ndarray, y: np.ndarray, y_pred: np.ndarray, idx_to_class: dict):
    max_images = min(64, x.shape[0])
    num_cols = min(8, max_images)
    num_rows = max(1, ceil(max_images / num_cols))
    fig, axs = plt.subplots(num_rows, num_cols, dpi=300, constrained_layout=True, subplot_kw={"visible": False})

    for ax, image, label, scores in zip(axs.flat, x, y, y_pred):
        ax.set_visible(True)
        ax.axis("off")
        ax.imshow(image)

        # Encode labels and softmax scores in subplot title.
        prediction = np.argmax(scores)
        text = "{} {:.4f} ({})".format(idx_to_class[prediction], scores[prediction], idx_to_class[label])
        ax.set_title(text, {
            "fontsize": 2.5,
            "color": "green" if prediction == label else "red",
        })

    fig_image = render_figure(fig)

    return fig_image


def render_figure(fig: plt.Figure):
    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    image = np.array(canvas.buffer_rgba())

    return image


def grab_shuffled_data(dataloader: DataLoader, batch_size: int, args: Namespace):
    dataloader = DataLoader(dataloader.dataset, batch_size=64, shuffle=True, num_workers=args.num_workers)
    batch = next(iter(dataloader))

    return batch


def trainer_transform(x, y, y_pred, loss):
    return {"x": x, "y": y, "y_pred": y_pred, "loss": loss.item()}


def evaluator_transform(x, y, y_pred):
    return {"x": x, "y": y, "y_pred": y_pred}


def metric_transform(output):
    return output["y_pred"], output["y"]


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
    _args = parse_args()
    main(_args)
