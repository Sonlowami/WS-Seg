"""
Loss functions for the two encoder variants under test in Experiment 1.

Encoder I (intensity) and Encoder II (SDF) must NOT share a loss function --
using plain MSE for both would confound "different encoder" with "different,
weaker training signal" for Encoder II, undermining what Experiment 1 is
meant to isolate.
"""
import torch


def image_reconstruction_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Plain MSE, used for Encoder I (intensity targets, any channel count)."""
    return torch.mean((pred - target) ** 2)


def masked_eikonal_sdf_loss(pred_sdf: torch.Tensor, coords: torch.Tensor,
                             target_sdf: torch.Tensor, alpha: float,
                             eikonal_lambda: float) -> dict:
    """
    Equation 3, generalized to K SDF channels.

    pred_sdf, target_sdf: (N, K). coords: (N, 3) with requires_grad=True.

    MSE is taken over all points and channels. The Eikonal penalty is applied
    per channel, and only where that channel's own target satisfies
    |target| < alpha (the untruncated, near-boundary region).

    Why a loop over channels: autograd.grad on the full (N, K) output with
    ones as grad_outputs returns the gradient of the *sum* of channels, not of
    each channel. Each channel is a separate scalar field with its own unit
    gradient norm, so each needs its own backward pass. Cost is therefore K
    backward passes per step (with create_graph=True) for the Eikonal term.

    A channel that is entirely plateau (label absent in this case) has an
    all-zero near-boundary mask and contributes exactly zero Eikonal loss.

    Returns total plus the MSE and Eikonal components separately so they can
    be logged independently.
    """
    if pred_sdf.dim() == 1:
        pred_sdf = pred_sdf.unsqueeze(-1)
    if target_sdf.dim() == 1:
        target_sdf = target_sdf.unsqueeze(-1)

    mse = torch.mean((pred_sdf - target_sdf) ** 2)

    near_boundary = (target_sdf.abs() < alpha).float()          # (N, K)

    grad_norms = []
    for k in range(pred_sdf.shape[-1]):
        grad_k = torch.autograd.grad(
            outputs=pred_sdf[:, k].sum(), inputs=coords,
            create_graph=True, retain_graph=True,
        )[0]                                                     # (N, 3)
        grad_norms.append(grad_k.norm(dim=-1))
    grad_norm = torch.stack(grad_norms, dim=-1)                  # (N, K)

    eikonal_term = ((grad_norm - 1.0) ** 2) * near_boundary
    # If a batch has no near-boundary points at all, avoid dividing by zero.
    denom = near_boundary.sum().clamp(min=1.0)
    eikonal = eikonal_term.sum() / denom

    total = mse + eikonal_lambda * eikonal
    return {"total": total, "mse": mse.detach(), "eikonal": eikonal.detach()}
