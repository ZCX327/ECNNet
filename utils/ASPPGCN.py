import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class ASPPGraphFusion(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super(ASPPGraphFusion, self).__init__()
        mid_channels = mid_channels or out_channels // 2  # 每个分支输出减少一半

        # 五个不同尺度卷积分支
        self.branches = nn.ModuleList([
            nn.Conv2d(in_channels, mid_channels, kernel_size=1),
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, dilation=1),
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=2, dilation=2),
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=3, dilation=3),
            nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(in_channels, mid_channels, kernel_size=1)
            )
        ])

        # GCN：输入 mid_channels，输出 mid_channels
        self.gcn = GCNConv(mid_channels, mid_channels)

        # 注意力融合
        self.attn_weights = nn.Parameter(torch.ones(5))  # learnable attention

        # 通道融合压缩
        self.fusion = nn.Conv2d(mid_channels, out_channels, kernel_size=1)

    def forward(self, x):
        B, _, H, W = x.size()
        feats = []

        for i, branch in enumerate(self.branches):
            feat = branch(x)
            if i == 4:  # Global pooling分支
                feat = F.interpolate(feat, size=(H, W), mode='bilinear', align_corners=False)
            feats.append(feat)

        # 图结构建模：从每个分支提取图节点特征
        node_feats = torch.stack([f.mean(dim=(2, 3)) for f in feats], dim=1)  # [B, 5, C]

        # 构造全连接图
        edge_index = torch.tensor([
            [i for i in range(5) for j in range(5) if i != j],
            [j for i in range(5) for j in range(5) if i != j]
        ], dtype=torch.long, device=x.device)

        # 图卷积传播
        fused_nodes = torch.zeros_like(node_feats)
        for b in range(B):
            fused_nodes[b] = self.gcn(node_feats[b], edge_index)  # 每个样本独立图

        # 注意力加权融合
        weights = torch.softmax(self.attn_weights, dim=0)  # [5]
        out = 0
        for i in range(5):
            weight = fused_nodes[:, i].view(B, -1, 1, 1)  # [B, C, 1, 1]
            out += weights[i] * feats[i] * weight  # 每分支权重 + 节点特征调制

        return self.fusion(out)
