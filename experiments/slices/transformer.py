"""
SliceTransformer: predicts every slice's SDF decoder from the whole stack of
its volume's slice intensity decoders.

One sequence per (volume, modality, label): token t is slice t's intensity
decoder (standardized per coordinate with the training mean and std, as
HyperMLP's input), embedded by a Linear, plus a Fourier encoding of the
slice's relative position k / (n - 1) and a label embedding. A pre-LN,
bidirectional transformer encoder mixes the tokens; each token's output goes
through one zero-initialized Linear head to a residual on the label's base
decoder, as HyperMLP:

    decoder_t = base[label] + scale * head(h_t)

so training starts exactly at base (the floor). context=False is the
ablation: every token attends only to itself, so no information crosses
slices -- the same model and parameters, without neighbours.

The layers are built one by one (not nn.TransformerEncoder, whose deep copies
would start every layer with identical weights).
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

ACTIVATIONS = {"gelu": F.gelu, "relu": F.relu, "silu": F.silu}


def fourier_features(p: torch.Tensor, n_freq: int) -> torch.Tensor:
    """(...,) positions in [0, 1] -> (..., 2 n_freq + 1): p, sin and cos of
    2^j pi p, j < n_freq."""
    a = p[..., None] * (2.0 ** torch.arange(n_freq, device=p.device, dtype=p.dtype)) * math.pi
    return torch.cat([p[..., None], a.sin(), a.cos()], dim=-1)


class SliceTransformer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, n_labels: int = 1, d_model: int = 512, layers: int = 6,
                 heads: int = 4, activation: str = "gelu", dropout: float = 0.0, context: bool = True,
                 n_freq: int = 8):
        super().__init__()
        self.n_freq, self.context = n_freq, context
        self.embed = nn.Linear(in_dim, d_model)
        self.position = nn.Linear(2 * n_freq + 1, d_model)
        self.label = nn.Embedding(n_labels, d_model)
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model, heads, 4 * d_model, dropout, ACTIVATIONS[activation],
                                       batch_first=True, norm_first=True) for _ in range(layers)])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, out_dim)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)
        self.register_buffer("in_mean", torch.zeros(in_dim))
        self.register_buffer("in_std", torch.ones(in_dim))
        self.register_buffer("base", torch.zeros(n_labels, out_dim))
        self.register_buffer("scale", torch.ones(out_dim))

    def param_groups(self, weight_decay: float) -> list:
        """AdamW groups: weight decay on matrices (Linear and attention
        projections), not on biases, norms or the label embedding."""
        embedding = {id(self.label.weight)}
        decay = [p for p in self.parameters() if p.ndim >= 2 and id(p) not in embedding]
        ids = {id(p) for p in decay}
        return [{"params": decay, "weight_decay": weight_decay},
                {"params": [p for p in self.parameters() if id(p) not in ids], "weight_decay": 0.0}]

    def set_reference(self, inputs: torch.Tensor, base: torch.Tensor, scale: torch.Tensor):
        """As HyperMLP.set_reference: inputs (n_train, in_dim) training tokens."""
        self.in_mean.copy_(inputs.mean(0))
        self.in_std.copy_(inputs.std(0).clamp(min=1e-6) if len(inputs) > 1 else torch.ones_like(self.in_std))
        self.base.copy_(base)
        self.scale.copy_(scale)

    def forward(self, tokens: torch.Tensor, positions: torch.Tensor, label: torch.Tensor,
                padding: torch.Tensor = None) -> torch.Tensor:
        """tokens (B, T, in_dim), positions (B, T), label (B,), padding (B, T)
        True at padded tokens -> hidden states (B, T, d_model)."""
        h = (self.embed((tokens - self.in_mean) / self.in_std)
             + self.position(fourier_features(positions, self.n_freq)) + self.label(label)[:, None])
        mask = None
        if not self.context:
            # Self only (True = not attended). No padding mask then: no token
            # looks at another, and a padded token with every key masked would
            # give NaN attention, which 0 x NaN carries into real tokens.
            mask = ~torch.eye(tokens.shape[1], dtype=torch.bool, device=tokens.device)
            padding = None
        for layer in self.layers:
            h = layer(h, src_mask=mask, src_key_padding_mask=padding)
        return self.norm(h)

    def decoders(self, h: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        """h (N, d_model) hidden states, label (N,) -> (N, out_dim) flat decoders."""
        return self.base[label] + self.scale * self.head(h)


def sequence_predictor(net: SliceTransformer, items: list, device):
    """
    predict(chosen items) -> (len(chosen), out_dim) decoders, for items of
    one item list (dicts with case_id, modality, k, position, label_index
    and x, the intensity decoder with the position last). Each item's
    sequence is its (case_id, modality)'s slices in this list, ordered by k,
    run with the item's label; only the chosen tokens go through the head.
    Train and test lists build separate sequences, so a test sequence holds
    only its own volume's slices.
    """
    by_seq = {}
    for i in items:
        by_seq.setdefault((i["case_id"], i["modality"]), {})[i["k"]] = i
    seqs = {}
    for key, by_k in by_seq.items():
        ks = sorted(by_k)
        seqs[key] = (torch.stack([by_k[k]["x"][:-1] for k in ks]).to(device),
                     torch.tensor([by_k[k]["position"] for k in ks], dtype=torch.float32, device=device),
                     {k: t for t, k in enumerate(ks)})

    def predict(chosen: list) -> torch.Tensor:
        runs = sorted({((i["case_id"], i["modality"]), i["label_index"]) for i in chosen})
        length = max(len(seqs[s][2]) for s, _ in runs)
        dim = next(iter(seqs.values()))[0].shape[1]
        tokens = torch.zeros(len(runs), length, dim, device=device)
        positions = torch.zeros(len(runs), length, device=device)
        padding = torch.ones(len(runs), length, dtype=torch.bool, device=device)
        for b, (s, _) in enumerate(runs):
            n = len(seqs[s][2])
            tokens[b, :n], positions[b, :n], padding[b, :n] = seqs[s][0], seqs[s][1], False
        labels = torch.tensor([label for _, label in runs], device=device)
        h = net(tokens, positions, labels, padding)
        row = {run: b for b, run in enumerate(runs)}
        rows = [row[((i["case_id"], i["modality"]), i["label_index"])] for i in chosen]
        cols = [seqs[(i["case_id"], i["modality"])][2][i["k"]] for i in chosen]
        return net.decoders(h[rows, cols], torch.tensor([i["label_index"] for i in chosen], device=device))
    return predict


def print_parameter_summary(net: nn.Module, title: str) -> int:
    """Trainable parameters per top-level part and in total; returns the total."""
    total = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print(f"\n{title}: {total:,} trainable parameters")
    for name, child in net.named_children():
        print(f"  {name:<10} {sum(p.numel() for p in child.parameters()):>14,}")
    print(f"  fixed buffers (normalization, base decoders) {sum(b.numel() for b in net.buffers()):,}\n")
    return total
