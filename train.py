import torch
import random
from segment_anything import sam_model_registry
from mini_dataloader import BasicDataloader
from torch.utils.data import DataLoader
from utils import AverageMeter, cal_f1, cal_auc, cal_permute_f1, cal_image_level_acc
import numpy as np
from forensics_sam import ForensicsSAM
from adversary_detector import AdversaryDetector
from .utils.losses import GatedForensicsLoss

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
from tqdm import tqdm

import os
from PIL import Image

import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms


import os
from PIL import Image

import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from torchvision.transforms import InterpolationMode


class ForgeryDataset(Dataset):
    def __init__(self, root, image_size=1024):
        super().__init__()

        self.root = root

        self.au_dir = os.path.join(root, "AU")
        self.au_mask_dir = os.path.join(root, "AU_mask")

        self.tp_dir = os.path.join(root, "TP")
        self.tp_mask_dir = os.path.join(root, "TP_mask")

        self.samples = []

        valid_ext = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")

        # ---------------- Authentic ----------------
        for img_name in sorted(os.listdir(self.au_dir)):

            if not img_name.lower().endswith(valid_ext):
                continue

            base = os.path.splitext(img_name)[0]
            mask_name = base + "_gt.png"

            mask_path = os.path.join(self.au_mask_dir, mask_name)

            if os.path.exists(mask_path):

                self.samples.append({
                    "image": os.path.join(self.au_dir, img_name),
                    "mask": mask_path,
                    "label": 0
                })

        # ---------------- Tampered ----------------
        for img_name in sorted(os.listdir(self.tp_dir)):

            if not img_name.lower().endswith(valid_ext):
                continue

            base = os.path.splitext(img_name)[0]
            mask_name = base + "_gt.png"

            mask_path = os.path.join(self.tp_mask_dir, mask_name)

            if os.path.exists(mask_path):

                self.samples.append({
                    "image": os.path.join(self.tp_dir, img_name),
                    "mask": mask_path,
                    "label": 1
                })

        self.image_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ])

        self.mask_transform = transforms.Compose([
            transforms.Resize(
                (image_size, image_size),
                interpolation=InterpolationMode.NEAREST
            ),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):

        sample = self.samples[idx]

        image = Image.open(sample["image"]).convert("RGB")
        mask = Image.open(sample["mask"]).convert("L")

        image = self.image_transform(image)

        mask = self.mask_transform(mask)

        # Binary mask
        mask = (mask > 0.5).float()

        label = torch.tensor(sample["label"], dtype=torch.float32)

        return image, mask.squeeze(0), label

from torch.utils.data import DataLoader

train_dataset = ForgeryDataset(
    root="./Dataset",
    image_size=1024,
)

train_loader = DataLoader(
    train_dataset,
    batch_size=4,
    shuffle=True,
    num_workers=8,
    pin_memory=True,
    drop_last=True,
)


def train_gated():
    model_type = ['vit_b', 'vit_l', 'vit_h']

    checkpoint = {
        'vit_b': './weight/sam_vit_b_01ec64.pth',
        'vit_l': './weight/sam_vit_l_0b3195.pth',
        'vit_h': './weight/sam_vit_h_4b8939.pth'
    }

    sam_type = model_type[2]
    r, image_size = 8, 1024
    normalize_type = 2

    sam, image_embedding_size = sam_model_registry[sam_type](image_size=image_size, checkpoint=checkpoint.get(sam_type))
    model = ForensicsSAM(sam, r).cuda()

    adv_detector = AdversaryDetector().cuda().eval()
    save_path = "./weight/adversary_detector.pth"
    adv_detector.load_detector(save_path)

    loss_fn = GatedForensicsLoss()
    opt = torch.optim.Adam(model.sam.mask_decoder.parameters(), lr=1e-4, weight_decay = 1e-4)

    scaler = torch.cuda.amp.GradScaler()

    for step in range(STEPS):
        model.train()

        for images, gt_masks, gt_label in tqdm(train_loader):
            opt.zero_grad()
            BATCH = images.shape[0]

            images, gt_masks = images.cuda(), gt_masks.cuda()
            gt_label = gt_label.cuda()

            with torch.no_grad():
                logits, feats = adv_detector(images)
                preds = torch.argmax(logits, dim=1)

            with torch.cuda.amp.autocast():

                mask_prediction, cls_prediction, gated_mask, auth_logit, iou_pred = model(images, preds)
                best_idx = iou_pred.argmax(dim=1)

                losses = loss_fn(gated_mask, mask_prediction, auth_logit, gt_masks, gt_label, best_token_idx=best_idx)

            
            scaler.scale(losses["total"]).backward()

            scaler.unscale_(opt)      # IMPORTANT for AMP
            torch.nn.utils.clip_grad_norm_(
                model.sam.mask_decoder.parameters(), 1.0
            )

            scaler.step(opt)
            scaler.update()

    return model