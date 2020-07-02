import numpy as np
import torch
import torch.nn as nn

from pytorch_lightning import LightningModule
# from pytorch_lightning.metrics import Accuracy
# from pytorch_lightning.metrics.functional import to_categorical
from torchvision.models import vgg16, vgg16_bn

from nemo.metrics import Accuracy


class PlainClassifier(nn.Module):
    def __init__(self, feature_extractor, num_features, num_classes):
        super().__init__()

        self.pool = nn.AdaptiveAvgPool2d((7, 7))
        self.feature_extractor = feature_extractor
        self.classifier = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(512, 32),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(32, num_classes),
        )

    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        x = torch.log_softmax(x, dim=1)

        return x


class Classifier(LightningModule):
    def __init__(self, feature_extractor, num_features, num_classes):
        super().__init__()

        self.pool = nn.AdaptiveAvgPool2d((7, 7))
        self.feature_extractor = feature_extractor
        self.classifier = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(512, 32),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(32, num_classes),
        )

        self.normalize = nn.Softmax(dim=1)
        self.criterion = nn.CrossEntropyLoss()
        self.accuracy = Accuracy()

    def forward(self, x, normalize=True):
        x = self.feature_extractor(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)

        if normalize:
            x = self.normalize(x)

        return x

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters())

    def training_step(self, batch, batch_idx):
        image, target = batch

        prediction = self(image, normalize=False)
        loss = self.criterion(prediction, target)

        self.accuracy.reset()
        self.accuracy.update((prediction, target))
        acc = self.accuracy.compute()

        log = {
            "loss/train": loss,
            "acc/train": acc,
        }

        progress_bar = {
            "train_acc": acc,
        }

        results = {
            "loss": loss,
            "log": log,
            "progress_bar": progress_bar,
        }

        return results

    def validation_step(self, batch, batch_idx):
        image, target = batch

        prediction = self(image, normalize=False)
        loss = self.criterion(prediction, target)

        self.accuracy.update((prediction, target), batch_idx)

        results = {
            "val_loss": loss,
        }

        return results

    def validation_epoch_end(self, outputs):
        mean_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        acc = self.accuracy.compute()

        log = {
            "loss/val": mean_loss,
            "acc/val": acc,
        }

        results = {
            "val_loss": mean_loss,
            "val_acc": acc,
            "log": log,
        }

        return results

    def test_step(self, batch, batch_idx):
        image, target = batch

        prediction = self(image, normalize=False)
        loss = self.criterion(prediction, target)

        self.accuracy.update((prediction, target), batch_idx)

        results = {
            "test_loss": loss,
        }

        return results

    def test_epoch_end(self, outputs):
        mean_loss = torch.stack([x["test_loss"] for x in outputs]).mean()
        acc = self.accuracy.compute()

        log = {
            "loss/test": mean_loss,
            "acc/test": acc,
        }

        results = {
            "test_loss": mean_loss,
            "test_acc": acc,
            "log": log,
        }

        return results

    def track_metrics(self, prediction, target):
        indices = torch.argmax(prediction, dim=1)
        correct = torch.eq(indices, target).view(-1)

        num_correct = torch.sum(correct).item()
        num_examples = correct.shape[0]

        return num_correct, num_examples


def initialize_feature_extractor():
    full_model = vgg16_bn(pretrained=True)
    full_model = vgg16(pretrained=True)
    feature_extractor = full_model.features
    num_features = full_model.classifier[0].in_features

    for param in feature_extractor.parameters():
        param.requires_grad = False

    return feature_extractor, num_features


def initialize_classifier(num_classes):
    feature_extractor, num_features = initialize_feature_extractor()
    # classifier = Classifier(feature_extractor, num_features, num_classes)
    classifier = PlainClassifier(feature_extractor, num_features, num_classes)

    return classifier
