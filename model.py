import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from skimage.segmentation import slic


class HypergraphConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.weight = nn.Parameter(torch.Tensor(in_channels, out_channels))
        self.bias = nn.Parameter(torch.Tensor(out_channels))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight)
        nn.init.zeros_(self.bias)

    def forward(self, x, H):
        batch_size, num_nodes, num_edges = H.shape

        # 计算节点度 D_v 和超边度 D_e
        D_v = H.sum(dim=2)  # (batch_size, num_nodes)
        D_e = H.sum(dim=1)  # (batch_size, num_edges)

        # **修正: 用 `diag_embed` 确保 D_v 和 D_e 变成对角矩阵**
        D_v = torch.diag_embed(D_v)  # (batch_size, num_nodes, num_nodes)
        D_e = torch.diag_embed(D_e)  # (batch_size, num_edges, num_edges)

        # **计算伪逆，避免奇异矩阵**
        D_v_inv_sqrt = torch.linalg.pinv(torch.sqrt(D_v) + 1e-6)
        D_e_inv = torch.linalg.pinv(D_e + 1e-6)

        # **修正矩阵乘法维度**
        H_T = H.transpose(1, 2)  # (batch_size, num_edges, num_nodes)
        H_norm = D_v_inv_sqrt @ H @ D_e_inv @ H_T @ D_v_inv_sqrt

        # **计算超图卷积**
        x = H_norm @ x @ self.weight + self.bias

        return x


class HypergraphTransformerLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.hypergraph_conv = HypergraphConv(d_model, d_model)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, X, H):
        X_hyper = self.hypergraph_conv(X, H)
        attn_output, _ = self.self_attn(X_hyper, X_hyper, X_hyper)
        X = X + self.dropout(attn_output)
        X = self.norm1(X)
        ff_output = self.linear2(self.dropout(F.relu(self.linear1(X))))
        X = X + self.dropout(ff_output)
        X = self.norm2(X)
        return X


import torch.nn.functional as F


class MultiScaleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv5 = nn.Conv2d(in_channels, out_channels, kernel_size=5, stride=2, padding=2)
        self.conv8 = nn.Conv2d(in_channels, out_channels, kernel_size=8, stride=3, padding=2)
        self.conv12 = nn.Conv2d(in_channels, out_channels, kernel_size=12, stride=4, padding=2)

    def forward(self, x):
        x5 = self.conv5(x)
        x8 = self.conv8(x)
        x12 = self.conv12(x)

        # 使用F.interpolate将输出调整为相同的空间大小（例如128x128）
        x5_resized = F.interpolate(x5, size=(x.shape[2], x.shape[3]), mode='bilinear', align_corners=False)
        x8_resized = F.interpolate(x8, size=(x.shape[2], x.shape[3]), mode='bilinear', align_corners=False)
        x12_resized = F.interpolate(x12, size=(x.shape[2], x.shape[3]), mode='bilinear', align_corners=False)

        # 将调整后的特征图拼接在一起
        return torch.cat([x5_resized, x8_resized, x12_resized], dim=1)


class HypergraphImageClassifier(nn.Module):
    def __init__(self, num_classes, num_segments=[100, 200, 300]):
        super().__init__()
        self.num_segments = num_segments
        self.cnn = nn.Sequential(
            MultiScaleConv(3, 64),
            nn.ReLU(),
            MultiScaleConv(192, 128),
            nn.ReLU(),
            MultiScaleConv(384, 256),
            nn.ReLU(),
            MultiScaleConv(768, 512),
            nn.ReLU(),
            MultiScaleConv(1536, 1024),
            nn.ReLU()
        )
        self.hg_conv1 = HypergraphConv(1024, 512)
        self.hg_conv2 = HypergraphConv(512, 256)
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        segments = []
        for n in self.num_segments:
            img = x.permute(0, 2, 3, 1).cpu().numpy()
            seg = [slic(img[i], n_segments=n, compactness=10) for i in range(img.shape[0])]
            segments.append(torch.tensor(seg, device=x.device))

        cnn_features = self.cnn(x)
        cnn_features = cnn_features.view(cnn_features.shape[0], cnn_features.shape[1], -1).permute(0, 2, 1)

        H = self._build_hypergraph(segments, cnn_features)
        hg_features = self.hg_conv1(cnn_features, H)
        hg_features = F.relu(hg_features)
        hg_features = self.hg_conv2(hg_features, H)

        logits = self.fc(hg_features.mean(dim=1))
        return logits

    def _build_hypergraph(self, segments, features):
        batch_size, num_nodes, _ = features.shape
        H = torch.zeros(batch_size, num_nodes, sum([seg.max() + 1 for seg in segments]), device=features.device)
        col_offset = 0
        for seg in segments:
            for i in range(batch_size):
                unique_segs = torch.unique(seg[i])  # 每个超像素的唯一标签
                for j, s in enumerate(unique_segs):
                    # 需要确保 `seg[i] == s` 会返回一个二维的布尔张量，索引时考虑的是 H * W 展开后的形状
                    # H[i] 为 (num_nodes, num_edges)，所以在这里应该对应超像素的每个节点（像素）
                    H[i, seg[i] == s, j + col_offset] = 1
            col_offset += len(unique_segs)
        return H


class bigCNNHypergraph(nn.Module):
    def __init__(self, in_channels, out_channels, num_classes, num_superpixels=100):
        super().__init__()
        self.num_superpixels = num_superpixels
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=9, padding=4)
        self.hypergraph_conv = HypergraphConv(out_channels, out_channels)
        self.fc = nn.Linear(out_channels, num_classes)

    def forward(self, x):
        batch_size, _, H, W = x.shape

        # **修正: 先提取特征，再 permute 和 reshape**
        features = self.conv(x)  # (batch_size, out_channels, H, W)
        features = features.permute(0, 2, 3, 1)  # (batch_size, H, W, out_channels)
        features = features.reshape(batch_size, -1, features.shape[-1])  # (batch_size, H*W, out_channels)

        hyperedge_maps = []
        for i in range(batch_size):
            img = x[i].permute(1, 2, 0).cpu().numpy()
            segments = slic(img, n_segments=self.num_superpixels, compactness=10)
            segments = torch.tensor(segments, device=x.device)

            num_nodes = H * W
            num_edges = segments.max() + 1
            H_matrix = torch.zeros(num_nodes, num_edges, device=x.device)
            for j in range(num_nodes):
                row, col = j // W, j % W
                H_matrix[j, segments[row, col]] = 1
            hyperedge_maps.append(H_matrix)

        hyperedge_maps = torch.stack(hyperedge_maps)
        features = self.hypergraph_conv(features, hyperedge_maps)
        global_features = features.mean(dim=1)
        logits = self.fc(global_features)

        return logits


if __name__ == "__main__":
    batch_size, in_channels, H, W = 2, 3, 128, 128
    num_classes = 10
    num_superpixels = 100

    # Device configuration (GPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = bigCNNHypergraph(in_channels=in_channels, out_channels=64, num_classes=num_classes,
                             num_superpixels=num_superpixels).to(device)

    x = torch.randn(batch_size, in_channels, H, W).to(device)  # Move input tensor to GPU

    output = model(x)
    print(output.shape)
