import argparse
import time
import torch

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.attention_infusion import XCAInfusionBlock


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def sync_if_needed(device):
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def estimate_attention_ops(
    attention_type,
    n_tokens,
    dim,
    num_heads,
    area_size,
    window_size,
    num_kv_heads,
    num_landmarks,
    segment_size,
    topk_landmarks,
):
    head_dim = dim // num_heads
    if attention_type == "xca":
        return num_heads * n_tokens * (head_dim ** 2)
    if attention_type == "area":
        n_areas = max(1, n_tokens // max(1, area_size * area_size))
        return num_heads * n_areas * n_areas * head_dim
    if attention_type == "local_window":
        tokens_per_window = window_size * window_size
        n_windows = max(1, n_tokens // max(1, tokens_per_window))
        return num_heads * n_windows * tokens_per_window * tokens_per_window * head_dim
    if attention_type == "linear":
        return num_heads * n_tokens * (head_dim ** 2)
    if attention_type == "gqa":
        return num_heads * n_tokens * n_tokens * head_dim
    if attention_type == "nystrom":
        m = min(num_landmarks, n_tokens)
        return num_heads * (n_tokens * m + m * m + m * n_tokens) * head_dim
    if attention_type == "scout":
        s = max(1, (n_tokens + segment_size - 1) // segment_size)
        return num_heads * s * s * head_dim
    if attention_type == "anna":
        m = min(num_landmarks, n_tokens)
        seg = max(1, (n_tokens + m - 1) // m)
        routed = min(topk_landmarks, m) * seg
        return num_heads * n_tokens * routed * head_dim
    if attention_type == "mva":
        tokens_per_window = window_size * window_size
        n_windows = max(1, n_tokens // max(1, tokens_per_window))
        local_ops = num_heads * n_windows * tokens_per_window * tokens_per_window * head_dim
        linear_ops = num_heads * n_tokens * (head_dim ** 2)
        return local_ops + linear_ops
    return 0


def benchmark_one(
    attention_type,
    device,
    batch,
    n_tokens,
    dim,
    num_heads,
    area_size,
    window_size,
    num_kv_heads,
    num_landmarks,
    segment_size,
    topk_landmarks,
    warmup,
    iters,
):
    block = XCAInfusionBlock(
        dim=dim,
        num_heads=num_heads,
        use_infusion=False,
        attention_type=attention_type,
        area_size=area_size,
        window_size=window_size,
        num_kv_heads=num_kv_heads,
        num_landmarks=num_landmarks,
        segment_size=segment_size,
        topk_landmarks=topk_landmarks,
    ).to(device)
    block.eval()

    x = torch.randn(batch, n_tokens, dim, device=device)

    with torch.no_grad():
        for _ in range(warmup):
            _ = block(x, noise_scale=0.0)
        sync_if_needed(device)

        start = time.perf_counter()
        for _ in range(iters):
            _ = block(x, noise_scale=0.0)
        sync_if_needed(device)
        elapsed = time.perf_counter() - start

    ms = (elapsed / iters) * 1000.0
    est_ops = estimate_attention_ops(
        attention_type,
        n_tokens,
        dim,
        num_heads,
        area_size,
        window_size,
        num_kv_heads,
        num_landmarks,
        segment_size,
        topk_landmarks,
    )
    return ms, est_ops


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--tokens", type=int, default=1024)
    parser.add_argument("--dim", type=int, default=256)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--area_size", type=int, default=2)
    parser.add_argument("--window_size", type=int, default=4)
    parser.add_argument("--num_kv_heads", type=int, default=2)
    parser.add_argument("--num_landmarks", type=int, default=64)
    parser.add_argument("--segment_size", type=int, default=16)
    parser.add_argument("--topk_landmarks", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=50)
    args = parser.parse_args()

    device = get_device()
    print(f"Device: {device}")
    print(
        f"Shape: batch={args.batch}, tokens={args.tokens}, dim={args.dim}, "
        f"heads={args.heads}, area_size={args.area_size}, window_size={args.window_size}, "
        f"num_kv_heads={args.num_kv_heads}, num_landmarks={args.num_landmarks}, "
        f"segment_size={args.segment_size}, topk_landmarks={args.topk_landmarks}"
    )

    for attention_type in ["xca", "area", "local_window", "linear", "gqa", "nystrom", "scout", "anna", "mva"]:
        ms, est_ops = benchmark_one(
            attention_type=attention_type,
            device=device,
            batch=args.batch,
            n_tokens=args.tokens,
            dim=args.dim,
            num_heads=args.heads,
            area_size=args.area_size,
            window_size=args.window_size,
            num_kv_heads=args.num_kv_heads,
            num_landmarks=args.num_landmarks,
            segment_size=args.segment_size,
            topk_landmarks=args.topk_landmarks,
            warmup=args.warmup,
            iters=args.iters,
        )
        print(f"{attention_type:>12} | avg_forward_ms={ms:8.3f} | est_attn_ops={est_ops}")


if __name__ == "__main__":
    main()
