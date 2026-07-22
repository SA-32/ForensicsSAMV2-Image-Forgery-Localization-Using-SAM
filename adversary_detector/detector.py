import torch
import torch.nn as nn
import torchvision.models as models
import torch.fft
import timm

class AdversaryDetector(nn.Module):
    def __init__(self, backbone='resnet18', pretrained=True):
        super().__init__()
        if backbone == 'resnet18':
            self.backbone = models.resnet18(pretrained=pretrained)
            in_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
        elif backbone == 'resnet50':
            self.backbone = models.resnet50(pretrained=pretrained)
            in_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
        elif backbone == 'convnext_tiny':
            self.backbone = timm.create_model('convnext_tiny', pretrained=True, num_classes=0, global_pool='avg')
            in_features = self.backbone.num_features
            self.backbone.fc = nn.Identity()
        else:
            raise ValueError("Unsupported backbone")

        # 投影到 256 维的对比学习空间（中间特征）
        self.feat_proj = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU()
        )

        # 最终分类器（正常 vs 对抗）
        self.classifier = nn.Linear(256, 2)

    def forward(self, x):
        # 特征提取
        raw_features = self.backbone(x)      # shape [B, in_features]
        features = self.feat_proj(raw_features)    # shape [B, 256]
        logits = self.classifier(features)         # shape [B, 2]

        return logits, features

    def save_detector(self, path):
        assert isinstance(self, nn.Module), "model 必须是 nn.Module 类型"
        torch.save(
            {k: v for k, v in self.state_dict().items()},
            path
        )

    def load_detector(self, path):
        state_dict = torch.load(path, map_location='cpu')
        if isinstance(self, (torch.nn.DataParallel, torch.nn.parallel.DistributedDataParallel)):
            self.module.load_state_dict(state_dict, strict=False)
        else:
            self.load_state_dict(state_dict, strict=False)

