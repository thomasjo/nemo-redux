import torch
import torch.nn as nn

from torchvision.models import vgg16, vgg16_bn  # noqa


class Classifier(nn.Module):
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


def initialize_feature_extractor():
    full_model = vgg16_bn(pretrained=True)
    # full_model = vgg16(pretrained=True)
    feature_extractor = full_model.features
    num_features = full_model.classifier[0].in_features

    for param in feature_extractor.parameters():
        param.requires_grad = False

    return feature_extractor, num_features


def initialize_classifier(num_classes):
    feature_extractor, num_features = initialize_feature_extractor()
    classifier = Classifier(feature_extractor, num_features, num_classes)

    return classifier
