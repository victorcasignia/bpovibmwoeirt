import torch
import torch.nn as nn
from einops import rearrange
from .attention_infusion import AttentionInfusionBlock

class ConvStem(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(in_channels, out_channels // 2, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(out_channels // 2),
            nn.GELU(),
            nn.Conv2d(out_channels // 2, out_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.GELU()
        )

    def forward(self, x):
        return self.proj(x)

class Downsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)
        self.norm = nn.LayerNorm(out_channels)

    def forward(self, x):
        # x: [B, L, D]
        B, L, D = x.shape
        H = W = int(L ** 0.5)
        x = rearrange(x, 'b (h w) d -> b d h w', h=H, w=W)
        x = self.proj(x)
        x = rearrange(x, 'b d h w -> b (h w) d')
        x = self.norm(x)
        return x

class MIN(nn.Module):
    def __init__(self, in_channels=3, num_classes=1000, stem_channels=64, 
                 depths=[2, 2, 6, 2], dims=[64, 128, 256, 512], 
                 infusion_stages=[False, False, True, True],
                 attention_type='area', num_heads=8, area_size=2, window_size=4,
                 num_kv_heads=2, num_landmarks=64, segment_size=16, topk_landmarks=4):
        super().__init__()
        
        self.stem = ConvStem(in_channels, stem_channels)
        
        self.stages = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        
        curr_dim = stem_channels
        for i in range(4):
            if i > 0:
                self.downsamples.append(Downsample(curr_dim, dims[i]))
                curr_dim = dims[i]
            else:
                if stem_channels != dims[0]:
                    self.downsamples.append(nn.Linear(stem_channels, dims[0]))
                    curr_dim = dims[0]
                else:
                    self.downsamples.append(nn.Identity())
            
            blocks = nn.ModuleList([
                AttentionInfusionBlock(
                    dim=curr_dim,
                    use_infusion=infusion_stages[i],
                    attention_type=attention_type,
                    num_heads=num_heads,
                    area_size=area_size,
                    window_size=window_size,
                    num_kv_heads=num_kv_heads,
                    num_landmarks=num_landmarks,
                    segment_size=segment_size,
                    topk_landmarks=topk_landmarks,
                )
                for _ in range(depths[i])
            ])
            self.stages.append(blocks)
            
        self.norm = nn.LayerNorm(curr_dim)
        self.head = nn.Linear(curr_dim, num_classes)
        
    def forward(self, x, noise_scale=1.0):
        # x: [B, C, H, W]
        x = self.stem(x)
        
        B, C, H, W = x.shape
        x = rearrange(x, 'b c h w -> b (h w) c')
        
        for i in range(4):
            x = self.downsamples[i](x)
            for block in self.stages[i]:
                x = block(x, noise_scale=noise_scale)
                
        x = self.norm(x)
        x = x.mean(dim=1) # Global average pooling
        x = self.head(x)
        
        return x

def min_s(**kwargs):
    # Remove depths and dims from kwargs if they exist to avoid multiple values error
    kwargs.pop('depths', None)
    kwargs.pop('dims', None)
    return MIN(depths=[2, 2, 6, 2], dims=[64, 128, 256, 512], **kwargs)
