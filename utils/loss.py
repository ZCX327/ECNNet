import numpy as np
import torch
from scipy.ndimage import zoom
import torch.nn as nn


class BCEDiceLoss(nn.Module):
    def __init__(self, bce_weight=0.5, smooth=1e-5):
        super(BCEDiceLoss, self).__init__()
        self.bce_weight = bce_weight
        self.bce = nn.BCEWithLogitsLoss()
        self.smooth = smooth

    def forward(self, preds, targets):
        # BCE Loss（不需要对 preds 做 sigmoid）
        bce = self.bce(preds, targets)

        # Dice Loss（需要 sigmoid）
        probs = torch.sigmoid(preds)
        num = targets.size(0)
        probs = probs.view(num, -1)
        targets = targets.view(num, -1)

        intersection = (probs * targets).sum(dim=1)
        dice = (2. * intersection + self.smooth) / (probs.sum(dim=1) + targets.sum(dim=1) + self.smooth)
        dice_loss = 1 - dice.mean()

        return self.bce_weight * bce + (1 - self.bce_weight) * dice_loss



##### Binary segmentation #####
class TverskyLoss_Binary(nn.Module):
    def __init__(self, alpha=0.7):
        super(TverskyLoss_Binary, self).__init__()
        self.alpha = alpha
        self.beta = 1 - self.alpha

    def _tversky_loss(self, score, target):
        target = target.float()
        smooth = 1.0

        TP = torch.sum(score * target)
        FP = torch.sum((1 - target) * score)
        FN = torch.sum(target * (1 - score))

        loss = (TP + smooth) / (TP + self.alpha*FP + self.beta*FN + smooth)

        return 1 - loss

    def forward(self, inputs, target, sigmoid=True):
        if sigmoid:
            inputs = torch.sigmoid(inputs)

        assert inputs.size() == target.size(), 'predict {} & target {} shape do not match'.format(inputs.size(), target.size())

        loss = self._tversky_loss(inputs[:, 0], target[:, 0])

        return loss

##### Multi-class segmentation #####
class TverskyLoss(nn.Module):
    def __init__(self, n_classes, alpha=0.7):
        super(TverskyLoss, self).__init__()
        self.n_classes = n_classes
        self.alpha = alpha
        self.beta = 1 - self.alpha

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _tversky_loss(self, score, target):
        target = target.float()
        smooth = 1.0

        TP = torch.sum(score * target)
        FP = torch.sum((1 - target) * score)
        FN = torch.sum(target * (1 - score))

        loss = (TP + smooth) / (TP + self.alpha*FP + self.beta*FN + smooth)

        return 1 - loss

    def forward(self, inputs, target, softmax=True):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        target = self._one_hot_encoder(target)
        assert inputs.size() == target.size(), 'predict {} & target {} shape do not match'.format(inputs.size(), target.size())
        loss = 0.0
        for i in range(0, self.n_classes):
            loss += self._tversky_loss(inputs[:, i], target[:, i])

        return loss / self.n_classes




import torch
import torch.nn as nn
import torch.nn.functional as F

def compute_boundary_weight_map(masks, kernel_size=3, w0=10.0, sigma=2.0):
    """
    生成边界权重图（纯 PyTorch 实现，无需 OpenCV/scipy）
    
    Args:
        masks: (B, 1, H, W) binary tensor, values in {0, 1}
        kernel_size: 用于膨胀的卷积核大小（默认 3x3）
        w0: 边界区域的最大额外权重（例如 10 表示边界像素 BCE 权重为 1+10=11）
        sigma: 高斯衰减控制参数（影响范围）

    Returns:
        weight_map: (B, 1, H, W) float tensor, >= 1.0
    """
    B, C, H, W = masks.shape
    assert C == 1, "Mask should have 1 channel"

    # 创建膨胀卷积核（用于模拟形态学膨胀）
    padding = kernel_size // 2
    kernel = torch.ones((1, 1, kernel_size, kernel_size), device=masks.device)

    # 膨胀操作：dilated = max_pool or conv with ones
    # 使用 unfold 或 conv，这里用 conv 更简洁
    dilated = F.conv2d(
        masks.float(),
        kernel,
        padding=padding,
        groups=1
    )
    dilated = (dilated > 0).float()  # 转回 binary

    # 边界 = 膨胀区域 - 原始区域
    boundary = (dilated - masks.float()).clamp(min=0)  # (B,1,H,W)

    # 计算每个像素到边界的欧氏距离（使用距离变换的近似）
    # 简化版：我们直接用高斯核对 boundary 做模糊，作为距离的代理
    # 构建高斯核
    g_kernel_size = int(2 * (3 * sigma) + 1)
    if g_kernel_size % 2 == 0:
        g_kernel_size += 1
    ax = torch.arange(-g_kernel_size // 2 + 1., g_kernel_size // 2 + 1., device=masks.device)
    xx, yy = torch.meshgrid(ax, ax, indexing='ij')
    gauss_kernel = torch.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    gauss_kernel = gauss_kernel / gauss_kernel.sum()
    gauss_kernel = gauss_kernel.view(1, 1, g_kernel_size, g_kernel_size)

    # 对 boundary 进行高斯模糊，得到“距离”的反比（越近值越大）
    dist_map = F.conv2d(
        boundary,
        gauss_kernel,
        padding=g_kernel_size // 2
    )

    # 权重 = 1 + w0 * exp(-dist^2 / (2*sigma^2))，但我们已经用高斯模糊近似了 exp 部分
    # 所以直接：weight = 1 + w0 * dist_map（因为 dist_map 已是高斯响应）
    weight_map = 1.0 + w0 * dist_map

    return weight_map


class BCEDiceBoundaryLoss(nn.Module):
    def __init__(self, 
                 bce_weight=0.6, 
                 dice_weight=0.4,
                 boundary_w0=5.0,
                 boundary_sigma=5.0,
                 smooth=1e-6):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.boundary_w0 = boundary_w0
        self.boundary_sigma = boundary_sigma
        self.smooth = smooth

    def forward(self, pred_logits, target):
        """
        Args:
            pred_logits: (B, 1, H, W) raw logits from model (no sigmoid)
            target: (B, 1, H, W) binary ground truth mask (0 or 1)

        Returns:
            total_loss: scalar tensor
            loss_dict: dict with 'bce', 'dice', 'total'
        """
        # --- Step 1: Compute boundary weight map ---
        with torch.no_grad():
            weight_map = compute_boundary_weight_map(
                target, 
                w0=self.boundary_w0, 
                sigma=self.boundary_sigma
            )  # (B,1,H,W)

        # --- Step 2: Weighted BCE Loss ---
        bce_per_pixel = F.binary_cross_entropy_with_logits(
            pred_logits, target.float(), reduction='none'
        )
        weighted_bce = (bce_per_pixel * weight_map).mean()

        # --- Step 3: Dice Loss (soft) ---
        pred_probs = torch.sigmoid(pred_logits)
        intersection = (pred_probs * target).sum()
        union = pred_probs.sum() + target.sum()
        dice_loss = 1.0 - (2.0 * intersection + self.smooth) / (union + self.smooth)

        # --- Step 4: Combine ---
        total_loss = self.bce_weight * weighted_bce + self.dice_weight * dice_loss

        return total_loss, {
            'bce': weighted_bce.item(),
            'dice': dice_loss.item(),
            'total': total_loss.item()
        }