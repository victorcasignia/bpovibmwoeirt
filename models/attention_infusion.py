import torch
import torch.nn as nn
import torch.nn.functional as F

class VarianceGate(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.proj = nn.Linear(dim, dim)
        self.temperature = nn.Parameter(torch.ones(1))

    def forward(self, x):
        # x: [B, L, D]
        logits = self.proj(x)
        return torch.sigmoid(logits / self.temperature)

class XCA(nn.Module):
    """ Cross-Covariance Attention (XCA) as used in XCiT """
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        q = q.transpose(-2, -1)
        k = k.transpose(-2, -1)
        v = v.transpose(-2, -1)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).permute(0, 3, 1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class AreaSparseAttention(nn.Module):
    """
    Area-based sparse attention for vision tokens.
    It pools Q/K/V into non-overlapping spatial areas and attends across areas,
    then broadcasts attended area features back to tokens.
    """
    def __init__(self, dim, num_heads=8, area_size=2, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        self.area_size = area_size
        self.head_dim = dim // num_heads
        if dim % num_heads != 0:
            raise ValueError(f"dim ({dim}) must be divisible by num_heads ({num_heads})")

        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def _infer_hw(self, token_count):
        side = int(token_count ** 0.5)
        if side * side != token_count:
            raise ValueError(f"Token count {token_count} is not a perfect square.")
        return side, side

    def _area_pool(self, tensor_2d):
        # tensor_2d: [B, Hh, C, H, W]
        B, Hh, C, H, W = tensor_2d.shape
        area = self.area_size
        if H % area != 0 or W % area != 0:
            raise ValueError(f"H ({H}) and W ({W}) must be divisible by area_size ({area}).")

        h_area = H // area
        w_area = W // area
        tensor_2d = tensor_2d.view(B, Hh, C, h_area, area, w_area, area)
        pooled = tensor_2d.mean(dim=(4, 6))  # [B, Hh, C, h_area, w_area]
        pooled = pooled.view(B, Hh, C, h_area * w_area)
        return pooled, h_area, w_area

    def forward(self, x):
        # x: [B, N, C]
        B, N, C = x.shape
        H, W = self._infer_hw(N)

        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B, heads, N, head_dim]
        q, k, v = qkv[0], qkv[1], qkv[2]

        q2d = q.transpose(-2, -1).reshape(B, self.num_heads, self.head_dim, H, W)
        k2d = k.transpose(-2, -1).reshape(B, self.num_heads, self.head_dim, H, W)
        v2d = v.transpose(-2, -1).reshape(B, self.num_heads, self.head_dim, H, W)

        q_area, h_area, w_area = self._area_pool(q2d)  # [B, heads, d, A]
        k_area, _, _ = self._area_pool(k2d)
        v_area, _, _ = self._area_pool(v2d)

        q_area = q_area.transpose(-2, -1)  # [B, heads, A, d]
        k_area = k_area.transpose(-2, -1)
        v_area = v_area.transpose(-2, -1)

        attn = (q_area @ k_area.transpose(-2, -1)) * self.scale  # [B, heads, A, A]
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        out_area = attn @ v_area  # [B, heads, A, d]
        out_area = out_area.transpose(-2, -1).reshape(B, self.num_heads, self.head_dim, h_area, w_area)

        out = out_area.repeat_interleave(self.area_size, dim=3).repeat_interleave(self.area_size, dim=4)
        out = out[:, :, :, :H, :W]
        out = out.reshape(B, self.num_heads, self.head_dim, N).transpose(-2, -1)
        out = out.reshape(B, N, C)

        out = self.proj(out)
        out = self.proj_drop(out)
        return out

class LocalWindowAttention(nn.Module):
    """
    Sparse local-window attention for vision tokens.
    Attention is computed only within non-overlapping windows.
    """
    def __init__(self, dim, num_heads=8, window_size=4, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        self.window_size = window_size
        self.head_dim = dim // num_heads
        if dim % num_heads != 0:
            raise ValueError(f"dim ({dim}) must be divisible by num_heads ({num_heads})")

        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def _infer_hw(self, token_count):
        side = int(token_count ** 0.5)
        if side * side != token_count:
            raise ValueError(f"Token count {token_count} is not a perfect square.")
        return side, side

    def forward(self, x):
        # x: [B, N, C]
        B, N, C = x.shape
        H, W = self._infer_hw(N)
        ws = self.window_size
        if H % ws != 0 or W % ws != 0:
            raise ValueError(f"H ({H}) and W ({W}) must be divisible by window_size ({ws}).")

        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # [B, heads, N, d]

        q = q.reshape(B, self.num_heads, H, W, self.head_dim)
        k = k.reshape(B, self.num_heads, H, W, self.head_dim)
        v = v.reshape(B, self.num_heads, H, W, self.head_dim)

        h_windows = H // ws
        w_windows = W // ws

        q = q.view(B, self.num_heads, h_windows, ws, w_windows, ws, self.head_dim)
        k = k.view(B, self.num_heads, h_windows, ws, w_windows, ws, self.head_dim)
        v = v.view(B, self.num_heads, h_windows, ws, w_windows, ws, self.head_dim)

        q = q.permute(0, 1, 2, 4, 3, 5, 6).reshape(B, self.num_heads, h_windows * w_windows, ws * ws, self.head_dim)
        k = k.permute(0, 1, 2, 4, 3, 5, 6).reshape(B, self.num_heads, h_windows * w_windows, ws * ws, self.head_dim)
        v = v.permute(0, 1, 2, 4, 3, 5, 6).reshape(B, self.num_heads, h_windows * w_windows, ws * ws, self.head_dim)

        attn = (q @ k.transpose(-2, -1)) * self.scale  # [B, heads, num_windows, ws^2, ws^2]
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        out = attn @ v
        out = out.reshape(B, self.num_heads, h_windows, w_windows, ws, ws, self.head_dim)
        out = out.permute(0, 1, 2, 4, 3, 5, 6).reshape(B, self.num_heads, H, W, self.head_dim)
        out = out.reshape(B, self.num_heads, N, self.head_dim)
        out = out.transpose(1, 2).reshape(B, N, C)

        out = self.proj(out)
        out = self.proj_drop(out)
        return out

class LinearAttention(nn.Module):
    """
    Kernelized linear attention (phi(x)=ELU(x)+1) with associative computation.
    """
    def __init__(self, dim, num_heads=8, qkv_bias=False, proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        if dim % num_heads != 0:
            raise ValueError(f"dim ({dim}) must be divisible by num_heads ({num_heads})")

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def _feature_map(self, x):
        return F.elu(x) + 1.0

    def forward(self, x):
        # x: [B, N, C]
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # [B, heads, N, d]

        q = self._feature_map(q)
        k = self._feature_map(k)

        kv = torch.einsum('bhnd,bhne->bhde', k, v)
        k_sum = k.sum(dim=2)  # [B, heads, d]
        z = 1.0 / (torch.einsum('bhnd,bhd->bhn', q, k_sum).unsqueeze(-1) + 1e-6)
        out = torch.einsum('bhnd,bhde->bhne', q, kv) * z

        out = out.transpose(1, 2).reshape(B, N, C)
        out = self.proj(out)
        out = self.proj_drop(out)
        return out

class GroupedQueryAttention(nn.Module):
    """
    Grouped Query Attention (GQA): fewer K/V heads than Q heads.
    """
    def __init__(self, dim, num_heads=8, num_kv_heads=2, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim ({dim}) must be divisible by num_heads ({num_heads})")
        if num_heads % num_kv_heads != 0:
            raise ValueError(f"num_heads ({num_heads}) must be divisible by num_kv_heads ({num_kv_heads})")

        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = dim // num_heads
        self.group_size = num_heads // num_kv_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.k_proj = nn.Linear(dim, self.num_kv_heads * self.head_dim, bias=qkv_bias)
        self.v_proj = nn.Linear(dim, self.num_kv_heads * self.head_dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        # x: [B, N, C]
        B, N, C = x.shape
        q = self.q_proj(x).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)  # [B, h, N, d]
        k = self.k_proj(x).reshape(B, N, self.num_kv_heads, self.head_dim).transpose(1, 2)  # [B, hk, N, d]
        v = self.v_proj(x).reshape(B, N, self.num_kv_heads, self.head_dim).transpose(1, 2)

        k = k.repeat_interleave(self.group_size, dim=1)  # [B, h, N, d]
        v = v.repeat_interleave(self.group_size, dim=1)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        out = attn @ v
        out = out.transpose(1, 2).reshape(B, N, C)
        out = self.proj(out)
        out = self.proj_drop(out)
        return out

class NystromAttention(nn.Module):
    """
    Nyström attention approximation using landmarks.
    """
    def __init__(self, dim, num_heads=8, num_landmarks=64, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        if dim % num_heads != 0:
            raise ValueError(f"dim ({dim}) must be divisible by num_heads ({num_heads})")
        self.num_landmarks = num_landmarks
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def _landmark_pool(self, x, m):
        # x: [B, heads, N, d]
        B, H, N, D = x.shape
        if N % m == 0:
            seg = N // m
            return x.view(B, H, m, seg, D).mean(dim=3)
        # fallback: interpolate-ish chunking by padding
        pad = m - (N % m)
        x_pad = F.pad(x, (0, 0, 0, pad))
        n_pad = N + pad
        seg = n_pad // m
        return x_pad.view(B, H, m, seg, D).mean(dim=3)

    def forward(self, x):
        # x: [B, N, C]
        B, N, C = x.shape
        m = min(self.num_landmarks, N)

        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # [B, heads, N, d]

        q_landmarks = self._landmark_pool(q, m)
        k_landmarks = self._landmark_pool(k, m)

        attn_q_km = torch.softmax((q @ k_landmarks.transpose(-2, -1)) * self.scale, dim=-1)      # [B,h,N,m]
        attn_qm_km = torch.softmax((q_landmarks @ k_landmarks.transpose(-2, -1)) * self.scale, dim=-1)  # [B,h,m,m]
        attn_qm_k = torch.softmax((q_landmarks @ k.transpose(-2, -1)) * self.scale, dim=-1)      # [B,h,m,N]

        attn_q_km = self.attn_drop(attn_q_km)
        attn_qm_km = self.attn_drop(attn_qm_km)
        attn_qm_k = self.attn_drop(attn_qm_k)

        eye = torch.eye(m, device=attn_qm_km.device, dtype=attn_qm_km.dtype)
        reg = eye.unsqueeze(0).unsqueeze(0).expand_as(attn_qm_km)
        inv_mid = torch.linalg.inv((attn_qm_km + 1e-4 * reg).contiguous())
        out = attn_q_km @ inv_mid @ (attn_qm_k @ v)  # [B,h,N,d]

        out = out.transpose(1, 2).reshape(B, N, C)
        out = self.proj(out)
        out = self.proj_drop(out)
        return out

class ScoutSegmentAttention(nn.Module):
    """
    SCOUT-inspired segment-compression attention (2025).
    Compresses token groups into segment landmarks, attends over segments,
    and broadcasts back to token resolution.
    """
    def __init__(self, dim, num_heads=8, segment_size=16, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        self.segment_size = segment_size
        self.head_dim = dim // num_heads
        if dim % num_heads != 0:
            raise ValueError(f"dim ({dim}) must be divisible by num_heads ({num_heads})")

        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def _segment_pool(self, x):
        # x: [B, heads, N, d]
        B, H, N, D = x.shape
        seg = self.segment_size
        pad = (seg - (N % seg)) % seg
        if pad > 0:
            x = F.pad(x, (0, 0, 0, pad))
        n_pad = x.shape[2]
        n_seg = n_pad // seg
        x_seg = x.view(B, H, n_seg, seg, D)
        x_pool = x_seg.mean(dim=3)
        return x_pool, n_seg, pad

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q_pool, n_seg, pad = self._segment_pool(q)
        k_pool, _, _ = self._segment_pool(k)
        v_pool, _, _ = self._segment_pool(v)

        attn = (q_pool @ k_pool.transpose(-2, -1)) * self.scale  # [B,h,S,S]
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        out_pool = attn @ v_pool  # [B,h,S,d]

        out = out_pool.unsqueeze(3).expand(-1, -1, -1, self.segment_size, -1).reshape(B, self.num_heads, n_seg * self.segment_size, self.head_dim)
        if pad > 0:
            out = out[:, :, :N, :]

        out = out.transpose(1, 2).reshape(B, N, C)
        out = self.proj(out)
        out = self.proj_drop(out)
        return out

class ANNAAttention(nn.Module):
    """
    ANNA-inspired approximate nearest-neighbor attention (2025).
    Routes each query to top-k landmark segments and attends only to routed keys.
    """
    def __init__(self, dim, num_heads=8, num_landmarks=64, topk_landmarks=4, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        if dim % num_heads != 0:
            raise ValueError(f"dim ({dim}) must be divisible by num_heads ({num_heads})")
        self.num_landmarks = num_landmarks
        self.topk_landmarks = topk_landmarks
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def _segmentize(self, x, m):
        # x: [B,h,N,d]
        B, H, N, D = x.shape
        seg = (N + m - 1) // m
        n_pad = seg * m
        pad = n_pad - N
        if pad > 0:
            x = F.pad(x, (0, 0, 0, pad))
        x_seg = x.view(B, H, m, seg, D)
        return x_seg, seg, pad

    def forward(self, x):
        B, N, C = x.shape
        m = min(self.num_landmarks, N)
        k_landmarks = min(self.topk_landmarks, m)

        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # [B,h,N,d]

        k_seg, seg_len, _ = self._segmentize(k, m)
        v_seg, _, _ = self._segmentize(v, m)
        k_centroids = k_seg.mean(dim=3)  # [B,h,m,d]

        route_scores = (q @ k_centroids.transpose(-2, -1)) * self.scale  # [B,h,N,m]
        topk_idx = route_scores.topk(k=k_landmarks, dim=-1).indices  # [B,h,N,k]

        k_seg_exp = k_seg.unsqueeze(2).expand(-1, -1, N, -1, -1, -1)  # [B,h,N,m,seg,d]
        v_seg_exp = v_seg.unsqueeze(2).expand(-1, -1, N, -1, -1, -1)

        gather_idx = topk_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, -1, seg_len, self.head_dim)
        sel_k = torch.gather(k_seg_exp, 3, gather_idx).reshape(B, self.num_heads, N, k_landmarks * seg_len, self.head_dim)
        sel_v = torch.gather(v_seg_exp, 3, gather_idx).reshape(B, self.num_heads, N, k_landmarks * seg_len, self.head_dim)

        q_exp = q.unsqueeze(-2)  # [B,h,N,1,d]
        attn = (q_exp @ sel_k.transpose(-2, -1)).squeeze(-2) * self.scale  # [B,h,N,K]
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        out = (attn.unsqueeze(-2) @ sel_v).squeeze(-2)  # [B,h,N,d]
        out = out.transpose(1, 2).reshape(B, N, C)
        out = self.proj(out)
        out = self.proj_drop(out)
        return out

class MVAAttention(nn.Module):
    """
    MVA-inspired mixed-frequency attention (2025):
    combines linear global mixing and local sparse window mixing.
    """
    def __init__(self, dim, num_heads=8, window_size=4, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.linear_attn = LinearAttention(dim=dim, num_heads=num_heads, qkv_bias=qkv_bias, proj_drop=proj_drop)
        self.local_attn = LocalWindowAttention(
            dim=dim,
            num_heads=num_heads,
            window_size=window_size,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
        )
        self.mix_logit = nn.Parameter(torch.tensor(0.0))

    def forward(self, x):
        alpha = torch.sigmoid(self.mix_logit)
        out_linear = self.linear_attn(x)
        out_local = self.local_attn(x)
        return alpha * out_local + (1.0 - alpha) * out_linear

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class AttentionInfusionBlock(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        mlp_ratio=4.,
        qkv_bias=False,
        drop=0.,
        attn_drop=0.,
        use_infusion=False,
        attention_type='area',
        area_size=2,
        window_size=4,
        num_kv_heads=2,
        num_landmarks=64,
        segment_size=16,
        topk_landmarks=4,
    ):
        super().__init__()
        self.dim = dim
        self.use_infusion = use_infusion
        self.attention_type = attention_type
        
        self.norm1 = nn.LayerNorm(dim)
        if attention_type == 'xca':
            self.attn = XCA(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        elif attention_type == 'area':
            self.attn = AreaSparseAttention(
                dim,
                num_heads=num_heads,
                area_size=area_size,
                qkv_bias=qkv_bias,
                attn_drop=attn_drop,
                proj_drop=drop,
            )
        elif attention_type == 'local_window':
            self.attn = LocalWindowAttention(
                dim,
                num_heads=num_heads,
                window_size=window_size,
                qkv_bias=qkv_bias,
                attn_drop=attn_drop,
                proj_drop=drop,
            )
        elif attention_type == 'linear':
            self.attn = LinearAttention(
                dim,
                num_heads=num_heads,
                qkv_bias=qkv_bias,
                proj_drop=drop,
            )
        elif attention_type == 'gqa':
            self.attn = GroupedQueryAttention(
                dim,
                num_heads=num_heads,
                num_kv_heads=num_kv_heads,
                qkv_bias=qkv_bias,
                attn_drop=attn_drop,
                proj_drop=drop,
            )
        elif attention_type == 'nystrom':
            self.attn = NystromAttention(
                dim,
                num_heads=num_heads,
                num_landmarks=num_landmarks,
                qkv_bias=qkv_bias,
                attn_drop=attn_drop,
                proj_drop=drop,
            )
        elif attention_type == 'scout':
            self.attn = ScoutSegmentAttention(
                dim,
                num_heads=num_heads,
                segment_size=segment_size,
                qkv_bias=qkv_bias,
                attn_drop=attn_drop,
                proj_drop=drop,
            )
        elif attention_type == 'anna':
            self.attn = ANNAAttention(
                dim,
                num_heads=num_heads,
                num_landmarks=num_landmarks,
                topk_landmarks=topk_landmarks,
                qkv_bias=qkv_bias,
                attn_drop=attn_drop,
                proj_drop=drop,
            )
        elif attention_type == 'mva':
            self.attn = MVAAttention(
                dim,
                num_heads=num_heads,
                window_size=window_size,
                qkv_bias=qkv_bias,
                attn_drop=attn_drop,
                proj_drop=drop,
            )
        else:
            raise ValueError(
                f"Unsupported attention_type: {attention_type}. Use 'area', 'local_window', 'linear', 'gqa', 'nystrom', 'scout', 'anna', 'mva', or 'xca'."
            )
        
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=nn.GELU, drop=drop)
        
        if self.use_infusion:
            self.variance_gate = VarianceGate(dim)
            
    def forward(self, x, noise_scale=1.0):
        # x: [B, L, D]
        shortcut = x
        
        # XCA Attention
        x = x + self.attn(self.norm1(x))
        
        # MLP
        x_mlp = self.mlp(self.norm2(x))
        
        # Infusion (applied to the MLP output before residual connection)
        if self.use_infusion and self.training and noise_scale > 0:
            # Predict variance based on local features (using the pre-MLP state as context)
            variance = self.variance_gate(x)
            # Sample Brownian noise
            noise = torch.randn_like(x_mlp)
            # Infuse noise scaled by variance and global noise_scale
            x_mlp = x_mlp + noise * variance * noise_scale
            
        return x + x_mlp


# Backward compatibility
XCAInfusionBlock = AttentionInfusionBlock
