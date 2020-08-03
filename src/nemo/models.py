import torch
import torch.nn as nn
import torch.optim as optim

from pytorch_lightning import LightningModule
from torchvision.models import vgg16_bn


class Classifier(LightningModule):
    def __init__(self, feature_extractor, num_features, num_classes):
        super().__init__()

        self.criterion = nn.NLLLoss()

        self.feature_extractor = feature_extractor
        self.pool = nn.AdaptiveAvgPool2d((7, 7))

        self.classifier = nn.Sequential(
            # fc1
            nn.Linear(num_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            # fc2
            nn.Linear(512, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            # predictions
            nn.Linear(32, num_classes),
        )

        # Disable fine-tuning by default.
        self.requires_finetuning(False)

    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        x = torch.log_softmax(x, dim=1)

        return x

    def train(self, mode=True):
        super().train(mode)

        # Only enable feature extractor training mode when fine-tuning.
        finetuning_mode = self.finetuning and mode
        self.feature_extractor.train(finetuning_mode)

        return self

    def requires_finetuning(self, finetuning=True):
        self.finetuning = finetuning

        all_modules = list(self.feature_extractor.children())
        for module in all_modules:
            module.requires_grad_(False)
            module.train(False)

        finetune_modules = all_modules[-20:]
        for module in finetune_modules:
            module.requires_grad_(self.finetuning)
            module.train(self.finetuning)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-5)

        return optimizer

    def predict_on_batch(self, batch, batch_idx, prefix=None):
        image, target = batch

        prediction = self(image)
        loss = self.criterion(prediction, target)

        num_correct = torch.sum(prediction.argmax(dim=1) == target)
        num_examples = torch.tensor(target.shape[0])
        accuracy = torch.true_divide(num_correct, num_examples)

        log = {
            f"{prefix}/loss": loss,
            f"{prefix}/acc": accuracy,
        }

        return loss, accuracy, log

    def training_step(self, batch, batch_idx):
        loss, accuracy, log = self.predict_on_batch(batch, batch_idx, prefix="train")

        progress_bar = {"train_acc": accuracy}

        return {
            "loss": loss,
            "log": log,
            "progress_bar": progress_bar,
        }

    def validation_step(self, batch, batch_idx):
        loss, accuracy, log = self.predict_on_batch(batch, batch_idx, prefix="val")

        return {
            "val_loss": loss,
            "val_acc": accuracy,
            "log": log,
        }

    def validation_epoch_end(self, outputs):
        mean_loss = torch.stack([out["val_loss"] for out in outputs]).mean()
        mean_accuracy = torch.stack([out["val_acc"] for out in outputs]).mean()

        log = {
            "val/loss": mean_loss,
            "val/acc": mean_accuracy,
        }

        return {
            "val_loss": mean_loss,
            "val_acc": mean_accuracy,
            "log": log,
        }

    def test_step(self, batch, batch_idx):
        loss, accuracy, log = self.predict_on_batch(batch, batch_idx, prefix="test")

        return {
            "test_loss": loss,
            "test_acc": accuracy,
            "log": log,
        }

    def test_epoch_end(self, outputs):
        mean_loss = torch.stack([out["test_loss"] for out in outputs]).mean()
        mean_accuracy = torch.stack([out["test_acc"] for out in outputs]).mean()

        log = {
            "test/loss": mean_loss,
            "test/acc": mean_accuracy,
        }

        return {
            "log": log,
        }


def initialize_feature_extractor():
    full_model = vgg16_bn(pretrained=True)
    feature_extractor = full_model.features
    num_features = full_model.classifier[0].in_features

    return feature_extractor, num_features


def initialize_classifier(num_classes):
    feature_extractor, num_features = initialize_feature_extractor()
    classifier = Classifier(feature_extractor, num_features, num_classes)

    return classifier
