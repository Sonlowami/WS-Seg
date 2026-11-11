"""
The low-complexity hypernetwork: HyperMLP between PCA coordinates.

  input   an intensity decoder projected onto the top k_in principal
          directions of the training intensity decoders (k_in <= n - 1
          numbers instead of ~10^4 weights)
  output  coefficients on the top k_out principal directions of the
          training cases' ceiling decoders, one basis per label, turned back
          into decoder weights: w = origin[label] + c @ y_basis[label]

The origin is the same starting decoder as the full-weights hypernet's
base (the label's floor, or a fresh SIREN init), and HyperMLP's
zero-initialized last layer makes both networks start from exactly that
prediction. The basis is the origin -> mean ceiling decoder direction (so
the mean, what ridge regression collapsed to, is reachable at coefficient
1) followed by the principal directions, each scaled so that a coefficient
of 1 is one standard deviation of the training decoders along it. Without
an origin, the mean ceiling decoder is the origin and the first row is
dropped. Training is unchanged: the predicted weights run as the decoder
and are scored with the MSE + Eikonal SDF loss
(experiments/hypernet/train.train_weights).

Predictions are confined to origin + the span of the training ceiling
decoders' variation (and the mean): that is the bias that buys the low
complexity.
"""
import torch
import torch.nn as nn

from experiments.hypernet.model import HyperMLP


def pca_basis(X: torch.Tensor, k: int = None) -> tuple:
    """
    (mean (d,), basis (k, d), score std (k,)) of the rows of X, keeping
    k <= min(n - 1, rank) components (None = all of them).
    """
    X = X.double()
    n = X.shape[0]
    if n < 2:
        raise ValueError(f"PCA needs at least 2 rows, got {n}")
    mean = X.mean(0)
    _, S, Vh = torch.linalg.svd(X - mean, full_matrices=False)
    max_k = min(n - 1, int((S > S[0] * 1e-10).sum()))
    k = max_k if k is None else min(int(k), max_k)
    return mean.float(), Vh[:k].float(), (S[:k] / (n - 1) ** 0.5).float()


def output_bases(decoders_by_label: list, k: int = None, origins: torch.Tensor = None) -> tuple:
    """
    Per-label output bases from each label's training ceiling decoders
    (a list, in label order, of (n_l, p) tensors): (origin (L, p), y_basis
    (L, K, p)). Rows: with origins (L, p), first mean - origin, then the
    principal directions scaled by their score std; without, the origin is
    the mean and only the principal directions remain. Zero-padded to the
    largest K across labels.
    """
    bases = [pca_basis(Y, k) for Y in decoders_by_label]
    extra = 0 if origins is None else 1
    K = max(b[1].shape[0] for b in bases) + extra
    p = decoders_by_label[0].shape[1]
    y_basis = torch.zeros(len(bases), K, p)
    for j, (mean, basis, std) in enumerate(bases):
        if origins is not None:
            y_basis[j, 0] = mean - origins[j].to(mean)
        y_basis[j, extra:extra + len(std)] = basis * std[:, None]
    origin = torch.stack([b[0] for b in bases]) if origins is None else origins.detach().float().cpu()
    return origin, y_basis


class PCAHyperNet(nn.Module):
    """x (B, d) raw intensity decoder, label (B,) -> (B, p) SDF decoder weights,
    through HyperMLP on PCA coordinates. Same call signature as HyperMLP, so
    the training and evaluation code serve both."""

    def __init__(self, x_mean: torch.Tensor, x_basis: torch.Tensor, origin: torch.Tensor,
                 y_basis: torch.Tensor, **mlp_kwargs):
        super().__init__()
        self.register_buffer("x_mean", x_mean.clone())
        self.register_buffer("x_basis", x_basis.clone())          # (k_in, d)
        self.register_buffer("origin", origin.clone())            # (L, p) starting decoders
        self.register_buffer("y_basis", y_basis.clone())          # (L, k_out, p)
        n_labels, k_out = y_basis.shape[:2]
        self.hyper = HyperMLP(x_basis.shape[0], k_out, n_labels, **mlp_kwargs)

    def project(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.x_mean) @ self.x_basis.T

    def set_reference(self, inputs: torch.Tensor):
        """inputs: (n_train, d) training intensity decoders, for HyperMLP's
        standardization of the coordinates. Coefficients start at 0, i.e. at
        the origin decoders."""
        n_labels, k_out = self.y_basis.shape[:2]
        coords = self.project(inputs)
        self.hyper.set_reference(coords, torch.zeros(n_labels, k_out, device=coords.device),
                                 torch.ones(k_out, device=coords.device))

    def param_groups(self, weight_decay: float) -> list:
        return self.hyper.param_groups(weight_decay)

    def forward(self, x: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        coeff = self.hyper(self.project(x), label)                # (B, k_out)
        return self.origin[label] + torch.einsum("bk,bkp->bp", coeff, self.y_basis[label])
