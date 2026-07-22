from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn, Tensor

def dice_loss(pred, target, smooth=1e-5):
    pred = torch.sigmoid(pred)

    intersection = (pred * target).sum(dim=(1,2,3))
    union = pred.sum(dim=(1,2,3)) + target.sum(dim=(1,2,3))

    loss = 1 - (2 * intersection + smooth) / (union + smooth)
    return loss.mean()


class FocalLoss(nn.Module):

    def __init__(self, alpha: float = 0.75, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: Tensor, targets: Tensor) -> Tensor:
        p = torch.sigmoid(logits)
        ce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        p_t = p * targets + (1 - p) * (1 - targets)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        loss = alpha_t * (1 - p_t).pow(self.gamma) * ce
        return loss.mean()


class GatedForensicsLoss(nn.Module):
    
    def __init__(
        self,
        seg_weight: float = 1.0,
        cls_weight: float = 1.0,
        consistency_weight: float = 1.0,
        focal_alpha: float = 0.75,
        focal_gamma: float = 2.0,
    ):
        super().__init__()
        self.seg_weight = seg_weight
        self.cls_weight = cls_weight
        self.consistency_weight = consistency_weight
        self.seg_loss = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)

    def forward(
        self,
        gated_mask_logits: Tensor,   # B, K, H, W  (already gated -- this is what you deploy)
        raw_mask_logits: Tensor,     # B, K, H, W  (ungated -- used only for the consistency term)
        auth_logit: Tensor,          # B, 1
        gt_mask: Tensor,             # B, H, W     (all-zero for authentic samples)
        gt_label: Tensor,            # B           (1.0 = forged, 0.0 = authentic)
        best_token_idx: Optional[Tensor] = None,  # B  (which of the K mask tokens to score, e.g. argmax IoU)
    ):
        b, k, h, w = gated_mask_logits.shape
        if best_token_idx is None:
            best_token_idx = torch.zeros(b, dtype=torch.long, device=gated_mask_logits.device)

        idx = best_token_idx.view(b, 1, 1, 1).expand(-1, 1, h, w)
        gated_sel = torch.gather(gated_mask_logits, 1, idx).squeeze(1)  # B, H, W
        raw_sel = torch.gather(raw_mask_logits, 1, idx).squeeze(1)      # B, H, W

        gt_mask_resized = F.interpolate(gt_mask.unsqueeze(1).float(), size=(h, w), mode="nearest").squeeze(1)

        l_focal = self.seg_loss(gated_sel, gt_mask_resized)
        l_dice = dice_loss(gated_sel, gt_mask_resized)

        l_seg = l_focal + l_dice

        auth_logit_flat = auth_logit.view(b)
        l_cls = F.binary_cross_entropy_with_logits(auth_logit_flat, gt_label.float())

        pooled_raw = torch.logsumexp(raw_sel.flatten(1), dim=1)
        l_consistency = F.binary_cross_entropy_with_logits(pooled_raw, gt_label.float())

        total = self.seg_weight * l_seg + self.cls_weight * l_cls + self.consistency_weight * l_consistency

        return {
            "total": total,
            "seg": l_seg.detach(),
            "cls": l_cls.detach(),
            "consistency": l_consistency.detach(),
        }
