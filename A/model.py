import torch, torch.nn as nn, torch.nn.functional as F
from efficientnet_pytorch import EfficientNet

class EfficientNetClassifier(nn.Module):
    def __init__(self, num_classes, use_meta=False, meta_input_dim=0,
                 backbone_name="efficientnet-b4", dropout=0.5):
        super().__init__()
        self.use_meta = use_meta

        self.backbone = EfficientNet.from_pretrained(backbone_name)
        in_features = self.backbone._fc.in_features
        self.backbone._fc = nn.Identity()

        if self.use_meta and meta_input_dim > 0:
            self.meta_net = nn.Sequential(
                nn.Linear(meta_input_dim, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.3),
                nn.Linear(128, 64), nn.BatchNorm1d(64), nn.ReLU())
            in_features += 64
        else:
            self.meta_net = None

        self.classifier = nn.Sequential(
            nn.Linear(in_features, 256), nn.BatchNorm1d(256), nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )

    def forward(self, x, x_meta=None):
        x = self.backbone.extract_features(x)
        x = F.adaptive_avg_pool2d(x, 1).view(x.size(0), -1)
        if self.use_meta and self.meta_net is not None and x_meta is not None:
            x = torch.cat([x, self.meta_net(x_meta)], 1)
        return self.classifier(x)
