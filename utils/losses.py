import torch
import torch.nn as nn
import torch.nn.functional as F


def dice_loss(pred, target, smooth=1e-5):
    pred = torch.sigmoid(pred)

    intersection = (pred * target).sum(dim=(1,2,3))
    union = pred.sum(dim=(1,2,3)) + target.sum(dim=(1,2,3))

    loss = 1 - (2 * intersection + smooth) / (union + smooth)
    return loss.mean()


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.75, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, target):
        p = torch.sigmoid(logits)

        ce = F.binary_cross_entropy_with_logits(
            logits,
            target,
            reduction="none",
        )

        p_t = p * target + (1 - p) * (1 - target)
        alpha_t = self.alpha * target + (1 - self.alpha) * (1 - target)

        loss = alpha_t * (1 - p_t).pow(self.gamma) * ce

        return loss.mean()


class GatedForensicsLossv1(nn.Module):

    def __init__(self, raw_weight=1.0, gated_weight=1.0, cls_weight=1.0, focal_alpha=0.75, focal_gamma=2.0):
        super().__init__()

        self.raw_weight   = raw_weight
        self.gated_weight = gated_weight
        self.cls_weight   = cls_weight

        self.focal = FocalLoss(alpha = focal_alpha, gamma = focal_gamma)

    def segmentation_loss(self, logits, target):

        focal = self.focal(logits, target)
        dice = dice_loss(logits, target)

        return focal + dice

    def forward(self, gated_mask_logits, raw_mask_logits, auth_logit, gt_mask, gt_label):

        B, _, H, W = gated_mask_logits.shape

        gt_mask = F.interpolate(gt_mask.unsqueeze(1).float(), size=(H, W), mode="nearest")

        l_raw = self.segmentation_loss(raw_mask_logits, gt_mask)

        l_gated = self.segmentation_loss(gated_mask_logits, gt_mask)

        l_cls = F.binary_cross_entropy_with_logits(auth_logit.squeeze(1), gt_label.float())

        total = self.raw_weight * l_raw + self.gated_weight * l_gated + self.cls_weight * l_cls

        return {
            "total": total,
            "raw_seg": l_raw.detach(),
            "gated_seg": l_gated.detach(),
            "cls": l_cls.detach(),
        }

class GatedForensicsLossv2(nn.Module):

    def __init__(self,     seg_weight: float = 1.0, cls_weight: float = 1.0, consistency_weight: float = 1.0, focal_alpha: float = 0.75, focal_gamma: float = 2.0):
        super().__init__()

        self.seg_weight = seg_weight
        self.cls_weight = cls_weight
        self.consistency_weight = consistency_weight

        self.focal = FocalLoss(alpha = focal_alpha, gamma = focal_gamma)

    def segmentation_loss(self, logits, target):

        focal = self.focal(logits, target)
        dice = dice_loss(logits, target)

        return focal + dice

    def forward(self, gated_mask_logits, raw_mask_logits, auth_logit, gt_mask, gt_label):

        B, _, H, W = gated_mask_logits.shape

        gt_mask = F.interpolate(gt_mask.unsqueeze(1).float(), size=(H, W), mode="nearest")

        l_seg = self.segmentation_loss(gated_mask_logits, gt_mask)

        l_cls = F.binary_cross_entropy_with_logits(auth_logit.squeeze(1), gt_label.float())

        pooled_raw = torch.logsumexp(raw_mask_logits.flatten(1), dim=1)
        l_consistency = F.binary_cross_entropy_with_logits(pooled_raw, gt_label.float())

        total = self.seg_weight * l_seg + self.cls_weight * l_cls + self.consistency_weight * l_consistency

        return {
            "total": total,
            "seg": l_seg.detach(),
            "cls": l_cls.detach(),
            "consistency": l_consistency.detach(),
        }