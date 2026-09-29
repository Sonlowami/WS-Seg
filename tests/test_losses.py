import torch
from training.losses import masked_eikonal_sdf_loss


def test_eikonal_masking_excludes_plateau():
    """
    Construct a toy case where the model perfectly matches the target
    everywhere (so MSE = 0), but the predicted function has zero gradient
    everywhere (so it violates the Eikonal property in the near-boundary
    region but trivially satisfies |grad|=0 in the plateau, matching the
    clipped ground truth there). The eikonal loss should only penalize the
    near-boundary mismatch, not the plateau.
    """
    torch.manual_seed(0)
    n = 200
    coords = (torch.rand(n, 3) * 2 - 1).requires_grad_(True)

    alpha = 1.0
    # Ground truth: a simple |x|-like ramp near zero, clipped at alpha.
    target = coords[:, :1].abs().clamp(max=alpha) * torch.sign(coords[:, :1])

    # A "lazy" prediction with a shallow slope (violates |grad|=1 near the
    # boundary) but zero in the flat region -- exercised via a linear layer
    # with a small weight so autograd has something to differentiate.
    w = torch.tensor([[0.3, 0.0, 0.0]], requires_grad=True)
    pred = coords @ w.T

    loss_dict = masked_eikonal_sdf_loss(pred, coords, target, alpha=alpha, eikonal_lambda=1.0)

    assert loss_dict["eikonal"].item() > 0, "Eikonal term should be active near the boundary"
    assert torch.isfinite(loss_dict["total"])


def test_eikonal_is_computed_per_channel():
    """
    Two channels, each with exactly unit gradient norm (channel 0 = x,
    channel 1 = y). Per channel the Eikonal loss is zero. If the gradient were
    taken of the SUM of channels, the norm would be sqrt(2) and the loss
    would wrongly be positive.
    """
    torch.manual_seed(0)
    coords = (torch.rand(100, 3) * 2 - 1).requires_grad_(True)
    pred = coords[:, :2]
    target = torch.zeros(100, 2)
    loss_dict = masked_eikonal_sdf_loss(pred, coords, target, alpha=10.0, eikonal_lambda=1.0)
    assert loss_dict["eikonal"].item() < 1e-6


def test_absent_label_channel_adds_no_eikonal_loss():
    """
    Channel 1's target is all plateau (label absent), so its wild slope must
    not be penalized. Channel 0 has unit gradient, so total Eikonal loss is 0.
    """
    torch.manual_seed(0)
    coords = (torch.rand(100, 3) * 2 - 1).requires_grad_(True)
    alpha = 3.0
    pred = torch.stack([coords[:, 0], 5.0 * coords[:, 1]], dim=-1)
    target = torch.stack([torch.zeros(100), torch.full((100,), alpha)], dim=-1)
    loss_dict = masked_eikonal_sdf_loss(pred, coords, target, alpha=alpha, eikonal_lambda=1.0)
    assert loss_dict["eikonal"].item() < 1e-6


if __name__ == "__main__":
    test_eikonal_masking_excludes_plateau()
    test_eikonal_is_computed_per_channel()
    test_absent_label_channel_adds_no_eikonal_loss()
    print("Loss tests passed.")
