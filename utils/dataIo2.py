import torch
from pathlib import Path
from torchvision import transforms
import cv2
import numpy as np
import random


def read_img_any(path, to_rgb=True):
    """
    可读取 PNG / JPG / TIF（含16bit）的通用函数
    """
    path = str(path)

    # TIF 16bit fallback：尽量用 COLOR 读取
    if path.lower().endswith(('.tif', '.tiff')):
        img = cv2.imread(path, cv2.IMREAD_COLOR)  # 强制转为 RGB 模式
        if img is None:  # 实在读不了再 fallback
            raw = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            if raw is None:
                raise FileNotFoundError(path)
            if raw.dtype == np.uint16:
                raw = cv2.normalize(raw, None, 0, 255,
                                    cv2.NORM_MINMAX).astype(np.uint8)
            if len(raw.shape) == 2:
                img = cv2.cvtColor(raw, cv2.COLOR_GRAY2BGR)
            else:
                img = raw
    else:
        img = cv2.imread(path, cv2.IMREAD_COLOR)

    if img is None:
        raise FileNotFoundError(path)

    if to_rgb:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    return img


def read_mask_any(path, target_size=None):
    """
    掩膜读取（PNG/JPG/TIF 都可）
    强制输出单通道 uint8 (0/255)
    """
    path = str(path)

    # 读取灰度
    if path.lower().endswith(('.tif', '.tiff')):
        raw = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if raw is None:
            raise FileNotFoundError(path)

        if raw.dtype == np.uint16:
            raw = cv2.normalize(raw, None, 0, 255,
                                cv2.NORM_MINMAX).astype(np.uint8)

        if len(raw.shape) == 3:  # 多通道 → 灰度
            mask = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY)
        else:
            mask = raw
    else:
        mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

    if mask is None:
        raise FileNotFoundError(path)

    return mask


class ImageMaskDataset(torch.utils.data.Dataset):
    def __init__(self, dir_img, dir_mask, img_size=(512, 512), raw_Img=False):
        self.dir_img = Path(dir_img)
        self.dir_mask = Path(dir_mask)
        self.img_size = img_size
        self.raw_Img = raw_Img

        # 可支持的图像扩展名
        IMG_EXT = ('.jpg', '.jpeg', '.png', '.tif', '.tiff')

        self.pairs = []
        for img_path in self.dir_img.glob('*'):
            if img_path.suffix.lower() not in IMG_EXT:
                continue

            stem = img_path.stem

            # 尝试匹配 mask 的常见格式顺序
            for ext in ['.png', '.jpg', '.jpeg', '.tif']:
                mask_path = self.dir_mask / f"{stem}{ext}"
                if mask_path.exists():
                    self.pairs.append((img_path, mask_path))
                    break

        print(f"📌 Loaded {len(self.pairs)} image-mask pairs.")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img_path, mask_path = self.pairs[idx]

        # 读取 RGB 图像（兼容 TIF）
        img = read_img_any(img_path)

        # 读取 mask（可为 PNG/JPG/TIF/16bit）
        mask = read_mask_any(mask_path)

        # 尺寸缩放
        img = cv2.resize(img, self.img_size)
        mask = cv2.resize(mask, self.img_size, interpolation=cv2.INTER_NEAREST)

        # 随机翻转
        if random.random() > 0.5:
            img = cv2.flip(img, 1)
            mask = cv2.flip(mask, 1)

        # 图像→tensor + normalize
        img_tensor = transforms.ToTensor()(img)
        img_tensor = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )(img_tensor)

        # 掩膜二值化
        mask = (mask > 127).astype(np.float32)
        mask_tensor = torch.from_numpy(mask).float().unsqueeze(0)

        if self.raw_Img:
            return {"img_tensor": img_tensor,
                    "mask_tensor": mask_tensor,
                    "raw_img": img}
        else:
            return {"img_tensor": img_tensor,
                    "mask_tensor": mask_tensor}
