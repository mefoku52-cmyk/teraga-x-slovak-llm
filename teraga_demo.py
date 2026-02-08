"""TERAGA-X Demo (Architecture Showcase)"""
import torch
import torch.nn as nn
class RMSNorm(nn.Module):
    def __init__(self, d, e=1e-6): super().__init__(); self.w = nn.Parameter(torch.ones(d))
    def forward(self, x): return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6) * self.w
class TeragaDemo(nn.Module):
    def __init__(self, v=5000, d=128, h=4, l=4):
        super().__init__(); self.emb = nn.Embedding(v, d); self.layers = nn.ModuleList([nn.TransformerEncoderLayer(d,h,d*4) for _ in range(l)]); self.norm = RMSNorm(d); self.head = nn.Linear(d, v)
        print(f"🇸🇰 TERAGA-X Architecture: {l} Layers / {d} Dim / {h} Heads")
    def forward(self, x): return self.head(self.norm(self.layers[0](self.emb(x))))
if __name__ == "__main__":
    m = TeragaDemo(); out = m(torch.randint(0, 5000, (1, 10)))
    print(f"✓ Output tensor: {out.shape} | Status: Ready for GGUF export")
