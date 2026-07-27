import argparse
import time

import torch
import torch.nn as nn
import torch.nn.functional as F


def until(duration_s):
    end = time.time() + duration_s
    while time.time() < end:
        yield


def sync():
    torch.cuda.synchronize()


def transformer_inference(duration_s):
    device = "cuda"
    vocab = 8192
    d_model = 512
    seq = 128
    batch = 16
    emb = nn.Embedding(vocab, d_model, device=device)
    layer = nn.TransformerEncoderLayer(
        d_model=d_model,
        nhead=8,
        dim_feedforward=2048,
        batch_first=True,
        dropout=0.0,
        device=device,
    )
    model = nn.TransformerEncoder(layer, num_layers=4).eval()
    tokens = torch.randint(0, vocab, (batch, seq), device=device)
    with torch.no_grad():
        for _ in until(duration_s):
            x = emb(tokens)
            y = model(x)
            tokens = (y.argmax(dim=-1) + tokens) % vocab
    sync()


def pytorch_training(duration_s):
    device = "cuda"
    model = nn.Sequential(
        nn.Conv2d(3, 64, 3, padding=1),
        nn.ReLU(),
        nn.Conv2d(64, 128, 3, stride=2, padding=1),
        nn.ReLU(),
        nn.Conv2d(128, 128, 3, padding=1),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten(),
        nn.Linear(128, 10),
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    for _ in until(duration_s):
        x = torch.randn(96, 3, 128, 128, device=device)
        target = torch.randint(0, 10, (96,), device=device)
        opt.zero_grad(set_to_none=True)
        loss = F.cross_entropy(model(x), target)
        loss.backward()
        opt.step()
    sync()


def hpl_like(duration_s):
    device = "cuda"
    n = 2048
    eye = torch.eye(n, device=device)
    for _ in until(duration_s):
        a = torch.randn(n, n, device=device, dtype=torch.float32)
        a = a @ a.T + eye * 0.1
        b = torch.randn(n, 8, device=device, dtype=torch.float32)
        x = torch.linalg.solve(a, b)
        _ = (a @ x - b).norm()
    sync()


def cublas_gemm(duration_s):
    device = "cuda"
    a = torch.randn(4096, 4096, device=device)
    b = torch.randn(4096, 4096, device=device)
    c = torch.empty_like(a)
    for _ in until(duration_s):
        c = torch.mm(a, b, out=c)
        a, b = b, c
    sync()


def cudnn_convolution(duration_s):
    device = "cuda"
    conv = nn.Conv2d(64, 128, 3, padding=1).to(device)
    x = torch.randn(64, 64, 224, 224, device=device)
    with torch.no_grad():
        for _ in until(duration_s):
            x = conv(x)
            x = F.relu(x)
            x = F.conv2d(x, conv.weight.transpose(0, 1).contiguous(), padding=1)
            x = F.relu(x)
    sync()


def sha256_benchmark(duration_s):
    device = "cuda"
    x = torch.arange(1 << 22, device=device, dtype=torch.int64)
    k = torch.tensor(
        [
            0x428A2F98,
            0x71374491,
            0xB5C0FBCF,
            0xE9B5DBA5,
            0x3956C25B,
            0x59F111F1,
            0x923F82A4,
            0xAB1C5ED5,
        ],
        device=device,
        dtype=torch.int64,
    )
    for _ in until(duration_s):
        a = x
        b = x ^ 0x6A09E667
        c = x + 0xBB67AE85
        d = x ^ 0x3C6EF372
        for i in range(64):
            s1 = torch.bitwise_xor(torch.bitwise_right_shift(b, 6), torch.bitwise_left_shift(b, 26))
            ch = torch.bitwise_xor(torch.bitwise_and(b, c), torch.bitwise_and(torch.bitwise_not(b), d))
            t = a + s1 + ch + k[i & 7]
            a, b, c, d = d, t, a, c
        x = a ^ b ^ c ^ d
    sync()


def aes_benchmark(duration_s):
    device = "cuda"
    x = torch.arange(1 << 22, device=device, dtype=torch.int64)
    key = torch.full_like(x, 0x1F2E3D4C)
    for _ in until(duration_s):
        s = x ^ key
        for r in range(80):
            s = s ^ torch.bitwise_left_shift(s & 0x00FF00FF, 7)
            s = s ^ torch.bitwise_right_shift(s & 0xFF00FF00, 5)
            s = torch.bitwise_xor(torch.roll(s, shifts=1), s + (0x9E3779B9 + r))
        x = s
    sync()


def crc32_benchmark(duration_s):
    device = "cuda"
    x = torch.arange(1 << 23, device=device, dtype=torch.int64)
    poly = torch.tensor(0xEDB88320, device=device, dtype=torch.int64)
    crc = torch.full_like(x, -1)
    for _ in until(duration_s):
        v = x
        for _bit in range(32):
            mask = -(torch.bitwise_and(crc ^ v, 1))
            crc = torch.bitwise_xor(torch.bitwise_right_shift(crc, 1), torch.bitwise_and(poly, mask))
            v = torch.bitwise_right_shift(v, 1)
        x = crc ^ v
    sync()


def compression_benchmark(duration_s):
    device = "cuda"
    x = torch.randint(0, 256, (1 << 24,), device=device, dtype=torch.int16)
    for _ in until(duration_s):
        delta = x[1:] - x[:-1]
        small = torch.abs(delta) < 8
        packed = (delta.clamp(-8, 7) + 8).to(torch.int16)
        flags = small.to(torch.int16)
        score = torch.sum((packed ^ flags) & 0xF)
        x = torch.roll(x ^ score.to(torch.int16), shifts=257)
    sync()


MODES = {
    "vllm_inference_fallback": transformer_inference,
    "pytorch_training": pytorch_training,
    "hpl": hpl_like,
    "cublas_gemm": cublas_gemm,
    "cudnn_convolution": cudnn_convolution,
    "sha256": sha256_benchmark,
    "aes": aes_benchmark,
    "crc32": crc32_benchmark,
    "gpu_compression": compression_benchmark,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=sorted(MODES))
    parser.add_argument("--seconds", type=float, default=60.0)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available")
    torch.cuda.set_device(0)
    MODES[args.mode](args.seconds)


if __name__ == "__main__":
    main()
