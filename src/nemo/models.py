import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision as vision

from torchvision.models import vgg16_bn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor, TwoMLPHead
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor


class Classifier(nn.Module):
    def __init__(self, feature_extractor, num_features, num_classes):
        super().__init__()

        self.feature_extractor = feature_extractor
        self.pool = nn.AdaptiveAvgPool2d((7, 7))

        self.classifier = nn.Sequential(
            # fc1
            nn.Linear(num_features, 512, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.2),
            # fc2
            nn.Linear(512, 32, bias=False),
            nn.BatchNorm1d(32),
            nn.ReLU(),
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


class StochasticTwoMLPHead(TwoMLPHead):
    def __init__(self, in_channels, representation_size, dropout_rate=0.2):
        super().__init__(in_channels, representation_size)
        self.dropout_rate = dropout_rate

    def forward(self, x):
        x = x.flatten(start_dim=1)

        x = F.relu(self.fc6(x))
        x = F.dropout(x, self.dropout_rate)
        x = F.relu(self.fc7(x))
        x = F.dropout(x, self.dropout_rate)

        return x


def initialize_detector(num_classes, dropout_rate=0):
    # NOTE: See https://pytorch.org/docs/stable/torchvision/models.html#mask-r-cnn.
    model = vision.models.detection.maskrcnn_resnet50_fpn(
        pretrained=True,
        trainable_backbone_layers=5,
        box_detections_per_img=256,
    )

    # Use box head with dropout sampling support.
    model.roi_heads.box_head = StochasticTwoMLPHead(
        model.roi_heads.box_head.fc6.in_features,
        model.roi_heads.box_head.fc7.out_features,
        dropout_rate=dropout_rate,
    )

    # Customize box predictor.
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # Customize mask predictor.
    hidden_layer = 256
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layer, num_classes)

    return model
