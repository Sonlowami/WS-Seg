"""
Weight-space pieces of the hypernetwork experiment: a decoder head seen as
one flat vector, a functional forward that runs a head with given weights
(so the SDF loss backpropagates into whatever produced them), and the MLP
that maps an intensity decoder's vector to an SDF decoder's.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

ACTIVATIONS = {"gelu": nn.GELU, "relu": nn.ReLU, "silu": nn.SiLU}
NORMS = {"batch": nn.BatchNorm1d, "layer": nn.LayerNorm}


def hidden_block(in_dim: int, out_dim: int, activation: str = "gelu", norm: str = None,
                 dropout: float = 0.0) -> list:
    """Linear -> [norm] -> activation -> [dropout]: one hidden layer of the
    hypernetwork. norm: None, "batch" or "layer"."""
    if norm is not None and norm not in NORMS:
        raise ValueError(f"Unknown norm {norm!r}; use one of {sorted(NORMS)} or null")
    if not 0.0 <= dropout < 1.0:
        raise ValueError(f"dropout must be in [0, 1), got {dropout}")
    block = [nn.Linear(in_dim, out_dim)]
    if norm is not None:
        block.append(NORMS[norm](out_dim))
    block.append(ACTIVATIONS[activation]())
    if dropout > 0:
        block.append(nn.Dropout(dropout))
    return block


def flatten_state_dict(state_dict: dict) -> torch.Tensor:
    """A saved decoder (utils.io.save_decoder_weights) as one float vector."""
    return torch.cat([v.reshape(-1).float() for v in state_dict.values()])


class HeadLayout:
    """Parameter shapes of one decoder head, in head.parameters() order (the
    order of its state_dict), to split a flat vector back into tensors."""

    def __init__(self, head: nn.Module):
        self.shapes = [p.shape for p in head.parameters()]
        self.sizes = [p.numel() for p in head.parameters()]
        self.numel = sum(self.sizes)

    def unflatten(self, flat: torch.Tensor) -> list:
        return [chunk.reshape(shape) for chunk, shape in zip(flat.split(self.sizes), self.shapes)]

    def flatten(self, head: nn.Module) -> torch.Tensor:
        return torch.cat([p.detach().reshape(-1) for p in head.parameters()])

    def tensor_stds(self, head: nn.Module) -> torch.Tensor:
        """Each element gets the std of its own tensor in `head` (typically a
        freshly initialized one): the natural step size per tensor, since
        SIREN's first, hidden and last layers and the biases differ in scale
        by orders of magnitude. Biases get the std of PyTorch's default bias
        init, 1/sqrt(3 * fan_in), instead: a 1-channel output bias is a single
        number, whose sample std is NaN."""
        stds = []
        for layer in head:
            if isinstance(layer, nn.Linear):
                stds.append(float(layer.weight.detach().std().clamp(min=1e-6)))
                if layer.bias is not None:
                    stds.append((3 * layer.in_features) ** -0.5)
        return torch.cat([torch.full((n,), s) for s, n in zip(stds, self.sizes)])


def encode(model, coords: torch.Tensor) -> torch.Tensor:
    """The shared encoder of a STRAINER_INR."""
    out = coords
    for layer in model.encoder:
        out = layer(out)
    return out


def run_head(head: nn.Module, params: list, x: torch.Tensor) -> torch.Tensor:
    """head's forward with its Linear weights replaced by params (HeadLayout
    order); activations are the head's own."""
    params = iter(params)
    for layer in head:
        if isinstance(layer, nn.Linear):
            x = F.linear(x, next(params), next(params) if layer.bias is not None else None)
        else:
            x = layer(x)
    return x


def load_head(head: nn.Module, params: list):
    """Copy params into head, e.g. to predict a full volume with predict_volume."""
    with torch.no_grad():
        for p, value in zip(head.parameters(), params):
            p.copy_(value)


class HyperMLP(nn.Module):
    """
    (in_dim + n_labels) -> hidden_dim -> ... -> out_dim, `layers` Linear
    layers in total. The input is the intensity decoder's vector,
    standardized per coordinate with the training set's mean and std, joined
    with a one-hot of the label group to predict (sdf.per_label; a single
    label otherwise).

    Hidden layers are hidden_block(..., norm, dropout); the output layer is
    a plain Linear (no norm or dropout on the predicted weights).

    The output is a residual in units of each tensor's own scale:
    weights = base[label] + scale * mlp(x), where base[label] is a reference
    SDF decoder for that label (its floor, or a fresh SIREN init) and scale
    the per-tensor std (HeadLayout.tensor_stds). The last layer starts at
    zero, so training starts exactly at base. These references are buffers,
    saved with the state_dict.
    """

    def __init__(self, in_dim: int, out_dim: int, n_labels: int = 1, hidden_dim: int = 1024,
                 layers: int = 4, activation: str = "gelu", norm: str = None, dropout: float = 0.0):
        super().__init__()
        if layers < 2:
            raise ValueError(f"hypernet.layers must be >= 2, got {layers}")
        self.n_labels = n_labels
        dims = [in_dim + n_labels] + [hidden_dim] * (layers - 1)
        blocks = []
        for i in range(layers - 1):
            blocks += hidden_block(dims[i], dims[i + 1], activation, norm, dropout)
        blocks.append(nn.Linear(hidden_dim, out_dim))
        self.mlp = nn.Sequential(*blocks)
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)
        self.register_buffer("in_mean", torch.zeros(in_dim))
        self.register_buffer("in_std", torch.ones(in_dim))
        self.register_buffer("base", torch.zeros(n_labels, out_dim))
        self.register_buffer("scale", torch.ones(out_dim))

    def param_groups(self, weight_decay: float) -> list:
        """AdamW groups: weight decay on the Linear weights only, not on biases
        or norm parameters (decaying a norm's gain shrinks its layer's output,
        not its complexity). Decay pulls the weights, hence the residual,
        towards zero, i.e. the prediction towards base."""
        decay = [m.weight for m in self.mlp if isinstance(m, nn.Linear)]
        ids = {id(p) for p in decay}
        return [{"params": decay, "weight_decay": weight_decay},
                {"params": [p for p in self.parameters() if id(p) not in ids], "weight_decay": 0.0}]

    def set_reference(self, inputs: torch.Tensor, base: torch.Tensor, scale: torch.Tensor):
        """inputs: (n_train, in_dim) training vectors, for the standardization;
        base: (n_labels, out_dim), one reference decoder per label."""
        self.in_mean.copy_(inputs.mean(0))
        self.in_std.copy_(inputs.std(0).clamp(min=1e-6) if len(inputs) > 1 else torch.ones_like(self.in_std))
        self.base.copy_(base)
        self.scale.copy_(scale)

    def forward(self, x: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        """x: (B, in_dim); label: (B,) label indices."""
        onehot = F.one_hot(label, self.n_labels).to(x.dtype)
        h = torch.cat([(x - self.in_mean) / self.in_std, onehot], dim=-1)
        return self.base[label] + self.scale * self.mlp(h)
