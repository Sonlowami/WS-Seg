"""
AtlasResidual: a positional atlas and an image residual, trained jointly.

    weights = base[label] + scale * head(atlas(position, label) + image(x, label))

atlas and image are HyperMLP-style trunks (hidden_block layers), summed in
hidden space before one shared, zero-initialized output layer `head` -- so
training starts exactly at base, as HyperMLP, and the model has about the
hypernet's parameter count (one output layer; the atlas trunk is small
because its input is 1 + n_labels wide).

x is the hypernet's input, the intensity decoder's vector with the slice
position last; the atlas sees only that position. The image trunk also sees
the position, so its correction can depend on where the slice is.

image_dropout: in training, each item's image branch is dropped with this
probability (and the kept ones rescaled, as dropout). The atlas must then
predict on its own, so it learns the positional prior and the image branch
learns what the image adds on top -- without it, nothing stops the image
branch from absorbing the atlas too. At evaluation both branches are used.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from experiments.hypernet.model import hidden_block


class AtlasResidual(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, n_labels: int = 1, hidden_dim: int = 1024,
                 layers: int = 4, activation: str = "gelu", norm: str = None, dropout: float = 0.0,
                 image_dropout: float = 0.5):
        super().__init__()
        if layers < 2:
            raise ValueError(f"hypernet.layers must be >= 2, got {layers}")
        if not 0.0 <= image_dropout < 1.0:
            raise ValueError(f"image_dropout must be in [0, 1), got {image_dropout}")
        self.n_labels, self.image_dropout = n_labels, image_dropout

        def trunk(width):
            dims = [width] + [hidden_dim] * (layers - 1)
            return nn.Sequential(*[b for i in range(layers - 1)
                                   for b in hidden_block(dims[i], dims[i + 1], activation, norm, dropout)])

        # Built in HyperMLP's order (image trunk, then the output layer), so
        # under the same seed the image trunk starts as the hypernet's layers.
        self.image = trunk(in_dim + n_labels)
        self.head = nn.Linear(hidden_dim, out_dim)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)
        self.atlas = trunk(1 + n_labels)
        self.register_buffer("in_mean", torch.zeros(in_dim))
        self.register_buffer("in_std", torch.ones(in_dim))
        self.register_buffer("base", torch.zeros(n_labels, out_dim))
        self.register_buffer("scale", torch.ones(out_dim))

    def param_groups(self, weight_decay: float) -> list:
        """As HyperMLP: weight decay on the Linear weights only."""
        decay = [m.weight for m in self.modules() if isinstance(m, nn.Linear)]
        ids = {id(p) for p in decay}
        return [{"params": decay, "weight_decay": weight_decay},
                {"params": [p for p in self.parameters() if id(p) not in ids], "weight_decay": 0.0}]

    def set_reference(self, inputs: torch.Tensor, base: torch.Tensor, scale: torch.Tensor):
        """As HyperMLP.set_reference."""
        self.in_mean.copy_(inputs.mean(0))
        self.in_std.copy_(inputs.std(0).clamp(min=1e-6) if len(inputs) > 1 else torch.ones_like(self.in_std))
        self.base.copy_(base)
        self.scale.copy_(scale)

    def forward(self, x: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        """x: (B, in_dim), position last; label: (B,) label indices."""
        onehot = F.one_hot(label, self.n_labels).to(x.dtype)
        z = (x - self.in_mean) / self.in_std
        h = self.atlas(torch.cat([z[:, -1:], onehot], dim=-1))
        r = self.image(torch.cat([z, onehot], dim=-1))
        if self.training and self.image_dropout > 0:
            keep = (torch.rand(len(x), 1, device=x.device) >= self.image_dropout).to(r.dtype)
            r = r * keep / (1 - self.image_dropout)
        return self.base[label] + self.scale * self.head(h + r)
