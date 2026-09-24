"""
Feature Aggregation Network (FANet) for Industrial Anomaly Detection.
Combines multi-scale feature extraction with attention mechanisms and
memory-based comparison for high-accuracy defect identification.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision.models import EfficientNet_B0_Weights, ResNet18_Weights


class ChannelAttention(nn.Module):
    """Squeeze-and-Excitation channel attention module."""
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.shape
        avg_out = self.fc(self.avg_pool(x).view(b, c))
        max_out = self.fc(self.max_pool(x).view(b, c))
        attn = self.sigmoid(avg_out + max_out).view(b, c, 1, 1)
        return x * attn


class SpatialAttention(nn.Module):
    """Spatial attention module to focus on important regions."""
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        combined = torch.cat([avg_out, max_out], dim=1)
        attn = self.sigmoid(self.conv(combined))
        return x * attn


class CBAM(nn.Module):
    """Convolutional Block Attention Module combining channel + spatial attention."""
    def __init__(self, channels, reduction=16, spatial_kernel=7):
        super().__init__()
        self.channel_attn = ChannelAttention(channels, reduction)
        self.spatial_attn = SpatialAttention(spatial_kernel)

    def forward(self, x):
        x = self.channel_attn(x)
        x = self.spatial_attn(x)
        return x


class FeatureAggregationHead(nn.Module):
    """Aggregates multi-scale features into a unified representation."""
    def __init__(self, in_channels_list, out_dim=512):
        super().__init__()
        self.adaptors = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(c, 256, 1, bias=False),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                CBAM(256),
            )
            for c in in_channels_list
        ])
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        total = 256 * len(in_channels_list)
        self.projector = nn.Sequential(
            nn.Linear(total, out_dim, bias=False),
            nn.BatchNorm1d(out_dim),
            nn.ReLU(inplace=True),
            nn.Linear(out_dim, out_dim, bias=False),
            nn.BatchNorm1d(out_dim),
        )

    def forward(self, feature_list):
        pooled = []
        for feat, adaptor in zip(feature_list, self.adaptors):
            out = adaptor(feat)
            pooled.append(self.global_pool(out).flatten(1))
        agg = torch.cat(pooled, dim=1)
        return self.projector(agg)


class FANet(nn.Module):
    """
    Feature Aggregation Network for industrial anomaly detection.
    Uses EfficientNetB0 or ResNet18 as backbone with CBAM attention
    and multi-scale feature aggregation.
    """
    def __init__(self, backbone='efficientnet_b0', feature_dim=512, pretrained=True, num_classes=2):
        super().__init__()
        self.backbone_name = backbone
        self.feature_dim = feature_dim
        self.num_classes = num_classes

        if backbone == 'efficientnet_b0':
            weights = EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
            base = models.efficientnet_b0(weights=weights)
            self.layer0 = nn.Sequential(*list(base.features[:3]))
            self.layer1 = base.features[3]
            self.layer2 = base.features[4]
            self.layer3 = base.features[5]
            self.layer4 = nn.Sequential(base.features[6], base.features[7], base.features[8])
            in_channels_list = [40, 80, 112, 1280]
        else:
            weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
            base = models.resnet18(weights=weights)
            self.layer0 = nn.Sequential(base.conv1, base.bn1, base.relu, base.maxpool)
            self.layer1 = base.layer1
            self.layer2 = base.layer2
            self.layer3 = base.layer3
            self.layer4 = base.layer4
            in_channels_list = [64, 128, 256, 512]

        self.aggregation = FeatureAggregationHead(in_channels_list, out_dim=feature_dim)
        
        # Supervised Classification Head
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
        
        self._freeze_early_layers()

    def _freeze_early_layers(self, freeze=True):
        """Freeze/Unfreeze backbone layers."""
        for param in self.layer0.parameters(): param.requires_grad = not freeze
        for param in self.layer1.parameters(): param.requires_grad = not freeze
        for param in self.layer2.parameters(): param.requires_grad = not freeze
        for param in self.layer3.parameters(): param.requires_grad = not freeze
        for param in self.layer4.parameters(): param.requires_grad = not freeze
        # Keep aggregation and classifier always trainable
        for param in self.aggregation.parameters(): param.requires_grad = True
        for param in self.classifier.parameters(): param.requires_grad = True

    def forward(self, x):
        f0 = self.layer0(x)
        f1 = self.layer1(f0)
        f2 = self.layer2(f1)
        f3 = self.layer3(f2)
        f4 = self.layer4(f3)
        embedding = self.aggregation([f1, f2, f3, f4])
        norm_emb = F.normalize(embedding, dim=-1)
        logits = self.classifier(embedding)
        return norm_emb, logits

    def extract_feature_maps(self, x):
        """Return intermediate feature maps for heatmap generation."""
        f0 = self.layer0(x)
        f1 = self.layer1(f0)
        f2 = self.layer2(f1)
        f3 = self.layer3(f2)
        f4 = self.layer4(f3)
        embedding = self.aggregation([f1, f2, f3, f4])
        norm_emb = F.normalize(embedding, dim=-1)
        logits = self.classifier(embedding)
        return norm_emb, [f1, f2, f3, f4], logits


def build_fanet(backbone='efficientnet_b0', feature_dim=512, pretrained=True, num_classes=2):
    return FANet(backbone=backbone, feature_dim=feature_dim, pretrained=pretrained, num_classes=num_classes)
