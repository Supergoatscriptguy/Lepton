"""LEPTON - Lightweight Efficient Processing Transformer with Optimized Neurons"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.w = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.w


class SwiGLU(nn.Module):
    def __init__(self, dim, mult=4):
        super().__init__()
        hidden = ((int(dim * mult * 2 / 3) + 31) // 32) * 32
        self.w1 = nn.Linear(dim, hidden, bias=False)
        self.w2 = nn.Linear(hidden, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class ChunkedLinearAttention(nn.Module):
    """O(n) attention with chunking for memory efficiency."""
    def __init__(self, dim, num_heads=8, chunk_size=64):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.chunk_size = chunk_size
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.out = nn.Linear(dim, dim, bias=False)
        self.register_buffer('causal_mask', None)

    def _get_causal_mask(self, size, device):
        if self.causal_mask is None or self.causal_mask.shape[0] != size:
            self.causal_mask = torch.triu(torch.ones(size, size, device=device), diagonal=1).bool()
        return self.causal_mask

    def forward(self, x):
        B, N, C = x.shape
        H, D = self.num_heads, self.head_dim

        qkv = self.qkv(x).reshape(B, N, 3, H, D).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        pad = (self.chunk_size - N % self.chunk_size) % self.chunk_size
        if pad > 0:
            q, k, v = F.pad(q, (0,0,0,pad)), F.pad(k, (0,0,0,pad)), F.pad(v, (0,0,0,pad))

        N_padded = q.shape[2]
        num_chunks = N_padded // self.chunk_size

        q = q.reshape(B, H, num_chunks, self.chunk_size, D)
        k = k.reshape(B, H, num_chunks, self.chunk_size, D)
        v = v.reshape(B, H, num_chunks, self.chunk_size, D)

        # Within-chunk attention
        attn = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = attn.masked_fill(self._get_causal_mask(self.chunk_size, x.device), -65000.0)
        attn = F.softmax(attn, dim=-1)
        out_local = torch.matmul(attn, v)

        # Cross-chunk cumulative state
        if num_chunks > 1:
            chunk_k, chunk_v = k.mean(dim=3), v.mean(dim=3)
            cumsum_k = torch.cumsum(chunk_k, dim=2)
            cumsum_v = torch.cumsum(chunk_v, dim=2)
            zeros = torch.zeros(B, H, 1, D, device=x.device, dtype=x.dtype)
            cumsum_k = torch.cat([zeros, cumsum_k[:, :, :-1]], dim=2)
            cumsum_v = torch.cat([zeros, cumsum_v[:, :, :-1]], dim=2)

            cross_attn = torch.sigmoid(torch.einsum('bhcsd,bhcd->bhcs', q, cumsum_k) * self.scale)
            cross_v = cumsum_v.unsqueeze(3).expand(-1, -1, -1, self.chunk_size, -1)
            out = out_local + cross_attn.unsqueeze(-1) * cross_v * 0.5
        else:
            out = out_local

        out = out.reshape(B, H, N_padded, D)[:, :, :N].transpose(1, 2).reshape(B, N, C)
        return self.out(out)


class ThoughtTokens(nn.Module):
    """Internal reasoning tokens that help the model think."""
    def __init__(self, dim, num_thoughts=4):
        super().__init__()
        self.num_thoughts = num_thoughts
        self.thought_embeds = nn.Parameter(torch.randn(num_thoughts, dim) * 0.02)
        self.thought_proj = nn.Linear(dim, dim, bias=False)

    def forward(self, x):
        B, N, C = x.shape
        cond = self.thought_proj(x.mean(dim=1, keepdim=True))
        thoughts = self.thought_embeds.unsqueeze(0).expand(B, -1, -1) + cond
        return torch.cat([thoughts, x], dim=1), self.num_thoughts


class MemoryCrystal(nn.Module):
    """Persistent memory that survives across context."""
    def __init__(self, dim, num_slots=8):
        super().__init__()
        self.num_slots = num_slots
        self.dim = dim
        self.memory = nn.Parameter(torch.zeros(num_slots, dim))
        self.write_query = nn.Linear(dim, dim, bias=False)
        self.write_key = nn.Linear(dim, dim, bias=False)
        self.read_query = nn.Linear(dim, dim, bias=False)
        self.gate = nn.Linear(dim, 1, bias=False)

    def forward(self, x):
        B, N, C = x.shape
        mem = self.memory.unsqueeze(0).expand(B, -1, -1)

        # Read
        read_q = self.read_query(x)
        read_attn = F.softmax(torch.matmul(read_q, mem.transpose(-1, -2)) / math.sqrt(C), dim=-1)
        read_out = torch.matmul(read_attn, mem)
        x = x + torch.sigmoid(self.gate(x)) * read_out

        # Write
        write_q = self.write_query(mem)
        write_k = self.write_key(x)
        write_attn = F.softmax(torch.matmul(write_q, write_k.transpose(-1, -2)) / math.sqrt(C), dim=-1)
        new_content = torch.matmul(write_attn, x).mean(0)
        with torch.no_grad():
            self.memory.data = 0.9 * self.memory.data + 0.1 * new_content
        return x


class DepthRouter(nn.Module):
    """Routes tokens to different depths based on complexity."""
    def __init__(self, dim, num_depths=3):
        super().__init__()
        self.router = nn.Sequential(nn.Linear(dim, dim // 4), nn.GELU(), nn.Linear(dim // 4, num_depths))

    def forward(self, x):
        return F.softmax(self.router(x), dim=-1)


class FuturePeek(nn.Module):
    """Bidirectional hint about upcoming tokens."""
    def __init__(self, dim, peek_size=8):
        super().__init__()
        self.peek_size = peek_size
        self.compress = nn.Linear(dim * peek_size, dim, bias=False)
        self.gate = nn.Linear(dim * 2, dim, bias=False)

    def forward(self, x):
        B, N, C = x.shape
        padded = F.pad(x, (0, 0, 0, self.peek_size))
        peeks = padded[:, 1:].unfold(1, self.peek_size, 1).reshape(B, N, C * self.peek_size)
        peek_compressed = self.compress(peeks)
        gate = torch.sigmoid(self.gate(torch.cat([x, peek_compressed], dim=-1)))
        return x + gate * peek_compressed * 0.5


class SpeculativeHeads(nn.Module):
    """Predicts multiple tokens at once for faster generation."""
    def __init__(self, dim, vocab_size, num_speculative=4):
        super().__init__()
        self.heads = nn.ModuleList([nn.Linear(dim, vocab_size, bias=False) for _ in range(num_speculative)])
        self.confidence = nn.Sequential(nn.Linear(dim, dim // 4), nn.GELU(), nn.Linear(dim // 4, num_speculative), nn.Sigmoid())

    def forward(self, x):
        return [head(x) for head in self.heads], self.confidence(x)


class UncertaintyEstimator(nn.Module):
    """Tells you when the model is guessing vs confident."""
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, dim // 2), nn.GELU(), nn.Linear(dim // 2, 1), nn.Sigmoid())

    def forward(self, x):
        return self.net(x).squeeze(-1)


class LeptonBlock(nn.Module):
    def __init__(self, dim, num_heads=8, chunk_size=64):
        super().__init__()
        self.norm1 = RMSNorm(dim)
        self.attn = ChunkedLinearAttention(dim, num_heads, chunk_size)
        self.norm2 = RMSNorm(dim)
        self.mlp = SwiGLU(dim)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        return x + self.mlp(self.norm2(x))


class Lepton(nn.Module):
    def __init__(self, vocab_size=50257, dim=512, num_heads=8, num_layers=8, num_thoughts=4,
                 num_memory_slots=8, chunk_size=64, depths=[2, 4, 8], max_seq_len=2048, dropout=0.1, pad_token_id=1):
        super().__init__()
        self.dim = dim
        self.vocab_size = vocab_size
        self.num_thoughts = num_thoughts
        self.pad_token_id = pad_token_id
        self.max_seq_len = max_seq_len
        self.depths = depths

        self.tok_emb = nn.Embedding(vocab_size, dim, padding_idx=pad_token_id)
        self.drop = nn.Dropout(dropout)
        self.thoughts = ThoughtTokens(dim, num_thoughts)
        self.memory = MemoryCrystal(dim, num_memory_slots)
        self.peek = FuturePeek(dim, peek_size=8)
        self.router = DepthRouter(dim, len(depths))
        self.layers = nn.ModuleList([LeptonBlock(dim, num_heads, chunk_size) for _ in range(max(depths))])
        self.depth_norms = nn.ModuleList([RMSNorm(dim) for _ in depths])
        self.norm_out = RMSNorm(dim)
        self.lm_head = nn.Linear(dim, vocab_size, bias=False)
        self.speculative = SpeculativeHeads(dim, vocab_size, num_speculative=4)
        self.uncertainty = UncertaintyEstimator(dim)
        self.tok_emb.weight = self.lm_head.weight
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None: nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)
                if m.padding_idx is not None: m.weight.data[m.padding_idx].zero_()

    def forward(self, x, targets=None):
        B, N = x.shape
        h = self.drop(self.tok_emb(x))
        h, num_thoughts = self.thoughts(h)
        h = self.memory(h)
        h = self.peek(h)
        route_weights = self.router(h)

        outputs_at_depths = []
        for i, layer in enumerate(self.layers):
            h = layer(h)
            if (i + 1) in self.depths:
                outputs_at_depths.append(self.depth_norms[self.depths.index(i + 1)](h))

        if len(outputs_at_depths) > 1:
            stacked = torch.stack(outputs_at_depths, dim=-1)
            h = (stacked * route_weights.unsqueeze(2)).sum(dim=-1)
        else:
            h = outputs_at_depths[0]

        h = self.norm_out(h[:, num_thoughts:])
        logits = self.lm_head(h)
        spec_logits, spec_conf = self.speculative(h)
        uncertainty = self.uncertainty(h)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits[:, :-1].reshape(-1, self.vocab_size), targets[:, 1:].reshape(-1), ignore_index=self.pad_token_id)
            for i, sl in enumerate(spec_logits):
                shift = i + 2
                if N > shift:
                    loss = loss + 0.1 * F.cross_entropy(sl[:, :-shift].reshape(-1, self.vocab_size), targets[:, shift:].reshape(-1), ignore_index=self.pad_token_id)

        return {'logits': logits, 'loss': loss, 'speculative_logits': spec_logits, 'speculative_confidence': spec_conf, 'uncertainty': uncertainty}

    def _sample_token(self, logits, temperature, top_p):
        logits = logits / temperature
        sorted_logits, sorted_idx = torch.sort(logits, descending=True)
        cumsum = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        sorted_logits[cumsum - F.softmax(sorted_logits, dim=-1) > top_p] = -float('inf')
        return sorted_idx.gather(-1, torch.multinomial(F.softmax(sorted_logits, dim=-1), 1))

    @torch.no_grad()
    def generate(self, prompt_ids, max_new_tokens=100, temperature=0.8, top_p=0.9, use_speculative=True):
        self.eval()
        device = next(self.parameters()).device
        generated = torch.tensor([prompt_ids], device=device) if isinstance(prompt_ids, list) else prompt_ids.clone()
        return self._generate_speculative(generated, max_new_tokens, temperature, top_p) if use_speculative else self._generate_standard(generated, max_new_tokens, temperature, top_p)

    @torch.no_grad()
    def _generate_standard(self, generated, max_new_tokens, temperature, top_p):
        for _ in range(max_new_tokens):
            out = self.forward(generated[:, -self.max_seq_len:])
            next_tok = self._sample_token(out['logits'][:, -1], temperature, top_p)
            generated = torch.cat([generated, next_tok], dim=-1)
            if next_tok.item() == 0: break
        return generated[0].tolist()

    @torch.no_grad()
    def _generate_speculative(self, generated, max_new_tokens, temperature, top_p):
        tokens_generated = 0
        while tokens_generated < max_new_tokens:
            out = self.forward(generated[:, -self.max_seq_len:])
            next_tok = self._sample_token(out['logits'][:, -1], temperature, top_p)
            generated = torch.cat([generated, next_tok], dim=-1)
            tokens_generated += 1
            if next_tok.item() == 0 or tokens_generated >= max_new_tokens: break

            if out['uncertainty'][:, -1].item() < 0.5:
                for sl, conf in zip(out['speculative_logits'], out['speculative_confidence'][:, -1].squeeze()):
                    if tokens_generated >= max_new_tokens: break
                    if conf.item() > 0.7:
                        spec_tok = self._sample_token(sl[:, -1], temperature * 0.9, top_p)
                        generated = torch.cat([generated, spec_tok], dim=-1)
                        tokens_generated += 1
                        if spec_tok.item() == 0: break
                    else: break
        return generated[0].tolist()

    @torch.no_grad()
    def generate_with_uncertainty(self, prompt_ids, max_new_tokens=100, temperature=0.8, top_p=0.9):
        self.eval()
        device = next(self.parameters()).device
        generated = torch.tensor([prompt_ids], device=device) if isinstance(prompt_ids, list) else prompt_ids.clone()
        uncertainties = []
        for _ in range(max_new_tokens):
            out = self.forward(generated[:, -self.max_seq_len:])
            next_tok = self._sample_token(out['logits'][:, -1], temperature, top_p)
            generated = torch.cat([generated, next_tok], dim=-1)
            uncertainties.append(out['uncertainty'][:, -1].item())
            if next_tok.item() == 0: break
        return generated[0].tolist(), uncertainties

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# Model configs
def create_lepton_nano():
    return Lepton(dim=192, num_heads=3, num_layers=4, num_thoughts=1, num_memory_slots=2, chunk_size=32, depths=[2, 4])

def create_lepton_micro():
    return Lepton(dim=256, num_heads=4, num_layers=6, num_thoughts=2, num_memory_slots=4, chunk_size=32, depths=[2, 4, 6])

def create_lepton_tiny():
    return Lepton(dim=384, num_heads=6, num_layers=8, num_thoughts=3, num_memory_slots=6, chunk_size=64, depths=[2, 4, 8])

def create_lepton_small():
    return Lepton(dim=512, num_heads=8, num_layers=12, num_thoughts=4, num_memory_slots=8, chunk_size=64, depths=[3, 6, 12])

def create_lepton_medium():
    return Lepton(dim=768, num_heads=12, num_layers=16, num_thoughts=6, num_memory_slots=12, chunk_size=64, depths=[4, 8, 16])

def create_lepton_large():
    return Lepton(dim=1024, num_heads=16, num_layers=20, num_thoughts=8, num_memory_slots=16, chunk_size=64, depths=[5, 10, 20])

def create_lepton_xl():
    return Lepton(dim=1536, num_heads=24, num_layers=24, num_thoughts=12, num_memory_slots=24, chunk_size=64, depths=[6, 12, 24])

def create_lepton_xxl():
    return Lepton(dim=2048, num_heads=32, num_layers=32, num_thoughts=16, num_memory_slots=32, chunk_size=64, depths=[8, 16, 32])


if __name__ == '__main__':
    for name, fn in [("NANO", create_lepton_nano), ("MICRO", create_lepton_micro), ("TINY", create_lepton_tiny),
                     ("SMALL", create_lepton_small), ("MEDIUM", create_lepton_medium), ("LARGE", create_lepton_large),
                     ("XL", create_lepton_xl), ("XXL", create_lepton_xxl)]:
        print(f"LEPTON-{name}: {fn().count_parameters()/1e6:.1f}M")
