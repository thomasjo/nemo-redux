from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from pathlib import Path

import pytorch_lightning

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader, RandomSampler

from nemo.datasets import prepare_datasets
from nemo.models import initialize_classifier
from nemo.utils import ensure_reproducibility, ignore_warnings, timestamped_path


def create_trainer(args):
    early_stopping = EarlyStopping(monitor="val_loss", mode="min", patience=3)
    model_checkpoint = ModelCheckpoint(str(args.output_dir), monitor="val_loss", mode="min", save_top_k=3)
    logger = WandbLogger(save_dir=str(args.output_dir)) if not args.dev else None

    trainer = Trainer(
        gpus=args.num_gpus,
        fast_dev_run=args.dev,
        max_epochs=args.max_epochs,
        early_stop_callback=early_stopping,
        checkpoint_callback=model_checkpoint,
        # Configure logging...
        logger=logger,
        log_save_interval=1,
        row_log_interval=1,
        progress_bar_refresh_rate=1,
        weights_summary=None,
    )

    return trainer


def main(args):
    ignore_warnings(pytorch_lightning)
    ensure_reproducibility(seed=42)

    # Append timestamp to output directory.
    args.output_dir = timestamped_path(args.output_dir)
    args.output_dir.mkdir(parents=True)

    # TODO(thomasjo): Transition away from pre-partitioned datasets to on-demand partitioning.
    train_dataset, val_dataset, test_dataset = prepare_datasets(args.data_dir)
    num_classes = len(train_dataset.classes)

    sampling_factor = 4  # Oversample to get more random augmentations
    num_training_samples = sampling_factor * len(train_dataset)
    train_sampler = RandomSampler(train_dataset, replacement=True, num_samples=num_training_samples)

    train_dataloader = DataLoader(train_dataset, batch_size=32, sampler=train_sampler, num_workers=args.num_workers, pin_memory=True)
    val_dataloader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    model = initialize_classifier(num_classes)

    if args.dev:
        from torchsummary import summary
        summary(model, (3, 224, 224), train_dataloader.batch_size, device="cpu")

    trainer = create_trainer(args)
    trainer.fit(model, train_dataloader, val_dataloader)
    trainer.test(model, test_dataloader)


def parse_args():
    parser = ArgumentParser(formatter_class=lambda prog: ArgumentDefaultsHelpFormatter(prog, max_help_position=100))

    parser.add_argument("--data-dir", type=Path, metavar="PATH", required=True, help="path to partitioned data directory")
    parser.add_argument("--output-dir", type=Path, metavar="PATH", required=True, help="path to output directory")

    parser.add_argument("--num-gpus", type=int, metavar="NUM", default=1, help="number of GPUs to use for model training")
    parser.add_argument("--num-workers", type=int, metavar="NUM", default=2, help="number of workers to use for data loaders")

    parser.add_argument("--max-epochs", type=int, metavar="NUM", default=25, help="maximum number of epochs to train")
    parser.add_argument("--dev", action="store_true", help="run each model phase with only one batch")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
