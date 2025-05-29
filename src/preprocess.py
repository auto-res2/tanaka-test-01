import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset

def split_image_regions(img, patch_size=256, overlap=32):
    """
    Split an image (Tensor CxHxW) into overlapping patches.
    Returns a list of dicts with keys: 'patch' and 'coords' (y,x).
    """
    _, H, W = img.shape
    stride = patch_size - overlap
    regions = []
    for y in range(0, H - patch_size + 1, stride):
        for x in range(0, W - patch_size + 1, stride):
            patch = img[:, y:y+patch_size, x:x+patch_size]
            regions.append({'patch': patch, 'coords': (y, x)})
    return regions

class DummyDataset(Dataset):
    """
    A dummy dataset that returns the same image (and label) repeatedly.
    """
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]

def create_dummy_datasets():
    """Create dummy high-resolution datasets for testing."""
    standard_images = torch.rand(4, 3, 128, 128)  # standard resolution
    ultra_images = torch.rand(4, 3, 512, 512)     # ultra-high resolution
    labels = torch.randint(0, 10, (4,))
    
    standard_dataset = DummyDataset(standard_images, labels)
    ultra_dataset = DummyDataset(ultra_images, labels)
    
    return standard_dataset, ultra_dataset

class DummyPatchEmbedder(nn.Module):
    def __init__(self, in_dim, out_dim, bias=True):
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim, bias=bias)
    def forward(self, x):
        return self.fc(x)

class DummyTimestepEmbedder(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fc = nn.Linear(1, dim)
    def forward(self, t):
        t = t.unsqueeze(-1)  # (B,1)
        return self.fc(t)

class DummyLabelEmbedder(nn.Module):
    def __init__(self, num_classes, dim, dropout_prob):
        super().__init__()
        self.emb = nn.Embedding(num_classes, dim)
    def forward(self, y, training):
        return self.emb(y)

class DummyFiTBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, **kwargs):
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, int(hidden_size * mlp_ratio))
        self.fc2 = nn.Linear(int(hidden_size * mlp_ratio), hidden_size)
    def forward(self, x, c, mask, freqs_cos, freqs_sin, global_adaLNm):
        out = self.fc2(torch.relu(self.fc1(x)))
        return x + out

class DummyFinalLayer(nn.Module):
    def __init__(self, hidden_size, patch_size, out_channels, **kwargs):
        super().__init__()
        self.fc = nn.Linear(hidden_size, patch_size * patch_size * out_channels)
        self.patch_size = patch_size
        self.out_channels = out_channels
    def forward(self, x, c):
        B, N, D = x.shape
        out = self.fc(x)  # (B, N, patch_size**2 * out_channels)
        return out

class GatedRegionMerger(nn.Module):
    """
    A simple gated merging module that averages overlapping regions.
    """
    def __init__(self, in_channels, hidden_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(hidden_channels, in_channels, kernel_size=3, padding=1)
    
    def forward(self, region_results, output_size):
        B, C = region_results[0][0].shape[:2]
        H, W = output_size
        accum = torch.zeros(B, C, H, W, device=region_results[0][0].device)
        counts = torch.zeros_like(accum)
        patch_size = region_results[0][0].shape[-1]
        
        for patch_out, coords in region_results:
            y, x = coords
            accum[..., y:y+patch_size, x:x+patch_size] += patch_out
            counts[..., y:y+patch_size, x:x+patch_size] += 1
        merged = accum / counts.clamp(min=1.)
        gated = torch.relu(self.conv2(torch.relu(self.conv1(merged))))
        return gated

class FiT(nn.Module):
    """
    Dummy FiT model based on the research code.
    """
    def __init__(self,
                 context_size: int = 256,
                 patch_size: int = 4,
                 in_channels: int = 3,
                 hidden_size: int = 128,
                 depth: int = 4,
                 num_heads: int = 4,
                 num_classes: int = 10,
                 adaln_type: str = "lora",
                 **kwargs):
        super().__init__()
        self.context_size = context_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.learn_sigma = False  # simplified for dummy
        self.depth = depth
        
        self.x_embedder = DummyPatchEmbedder(in_channels * patch_size * patch_size, hidden_size)
        self.t_embedder = DummyTimestepEmbedder(hidden_size)
        self.y_embedder = DummyLabelEmbedder(num_classes, hidden_size, dropout_prob=0.1)
        
        self.blocks = nn.ModuleList([DummyFiTBlock(hidden_size, num_heads) for _ in range(depth)])
        self.final_layer = DummyFinalLayer(hidden_size, patch_size, in_channels)
    
    def forward(self, x, t, y, grid, mask, size=None):
        B = x.shape[0]
        x_flat = x.view(B, -1)  # (B, in_channels*patch_size**2)
        x_emb = self.x_embedder(x_flat).unsqueeze(1)  # (B, 1, hidden_size)
        t_emb = self.t_embedder(t)  # (B, hidden_size)
        y_emb = self.y_embedder(y, self.training)  # (B, hidden_size)
        c = t_emb + y_emb  # (B, hidden_size)
        x_feat = x_emb  # (B, 1, hidden_size)
        freqs_cos = torch.ones(B, 1, 1, device=x.device)
        freqs_sin = torch.ones(B, 1, 1, device=x.device)
        global_adaLNm = 0.0
        
        for block in self.blocks:
            x_feat = block(x_feat, c, mask, freqs_cos, freqs_sin, global_adaLNm)
        out = self.final_layer(x_feat, c)  # (B, 1, patch_size**2 * in_channels)
        out = out.view(B, self.in_channels, self.patch_size, self.patch_size)
        return out

    def finetune(self, type, unfreeze):
        if type == 'full':
            return
        for name, param in self.named_parameters():
            param.requires_grad = False
        for unf in unfreeze:
            for name, param in self.named_parameters():
                if unf in name:
                    param.requires_grad = True
