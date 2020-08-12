import torch
import torch.nn as nn

from torchvision.models import vgg16_bn


class Classifier(nn.Module):
    def __init__(self, feature_extractor, num_features, num_classes):
        super().__init__()

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

    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        x = torch.log_softmax(x, dim=1)

        return x


def initialize_feature_extractor():
    full_model = vgg16_bn(pretrained=True)
    num_features = full_model.classifier[0].in_features
    feature_extractor = full_model.features

    for param in feature_extractor.parameters():
        param.requires_grad = False

    return feature_extractor, num_features


def initialize_classifier(num_classes):
    feature_extractor, num_features = initialize_feature_extractor()
    classifier = Classifier(feature_extractor, num_features, num_classes)

    return classifier
