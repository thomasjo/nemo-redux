from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from datetime import datetime
from pathlib import Path

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.loggers.base import DummyLogger
from torch.utils.data import DataLoader, RandomSampler
from torchvision.utils import save_image  # noqa

from nemo.datasets import prepare_datasets
from nemo.models import initialize_classifier
from nemo.utils import ensure_reproducibility


def create_output_dir(output_dir):
    timestamp = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    output_dir = output_dir / timestamp
    output_dir.mkdir(parents=True)

    return output_dir


def main(args):
    ensure_reproducibility(seed=42)

    # TODO(thomasjo): Transition away from pre-partitioned datasets to on-demand partitioning.
    train_dataset, val_dataset, test_dataset = prepare_datasets(args.data_dir)
    num_classes = len(train_dataset.classes)

    # Use repeated sampling scheme (with "replacement") to better exploit random image augmentations.
    train_sampler = RandomSampler(train_dataset, replacement=True, num_samples=4 * len(train_dataset))
    train_dataloader = DataLoader(train_dataset, batch_size=32, sampler=train_sampler, num_workers=args.num_workers, pin_memory=True)
    val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    trainer = Trainer.from_argparse_args(args)
    trainer.weights_summary = None  # Disable annoying default

    output_dir = create_output_dir(args.output_dir)
    trainer.checkpoint_callback = ModelCheckpoint(filepath=str(output_dir))
    trainer.logger = WandbLogger(save_dir=str(output_dir)) if not args.fast_dev_run else None

    model = initialize_classifier(num_classes)
    trainer.fit(model, train_dataloader, val_dataloader)
    trainer.test(model, test_dataloader)


def parse_args():
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument("--data-dir", type=Path, required=True, help="path to partitioned data directory")
    parser.add_argument("--output-dir", type=Path, required=True, help="path to output root directory for logs, etc")

    parser.add_argument("--fast-dev-run", action="store_true", help="run each training phase with only one batch")
    parser.add_argument("--max-epochs", type=int, default=25, help="maximum number of epochs to train")

    parser.add_argument("--gpus", type=int, default=1, help="number of GPUs to use for training")
    parser.add_argument("--num-workers", type=int, default=2, help="number of workers to use for data loaders")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
