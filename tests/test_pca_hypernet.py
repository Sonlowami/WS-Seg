import torch

from experiments.hypernet.pca import PCAHyperNet, pca_basis, output_bases


def _decoders(n, p, seed):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(n, p, generator=g)


def test_pca_basis_caps_components_and_reconstructs():
    X = _decoders(6, 20, 0)
    mean, basis, std = pca_basis(X)
    assert basis.shape == (5, 20)                       # n - 1 components
    scores = (X - mean) @ basis.T
    assert torch.allclose(mean + scores @ basis, X, atol=1e-4)
    assert torch.allclose(scores.std(0), std, atol=1e-4)
    assert pca_basis(X, 50)[1].shape[0] == 5


def test_output_bases_are_padded_per_label():
    y_mean, y_basis = output_bases([_decoders(4, 10, 1), _decoders(7, 10, 2)])
    assert y_mean.shape == (2, 10) and y_basis.shape == (2, 6, 10)
    assert torch.count_nonzero(y_basis[0, 3:]) == 0     # label 0 has only 3 components


def test_starts_at_the_mean_decoder_and_spans_the_training_set():
    X, Y = _decoders(5, 30, 3), _decoders(5, 12, 4)
    x_mean, x_basis, _ = pca_basis(X)
    y_mean, y_basis = output_bases([Y])
    net = PCAHyperNet(x_mean, x_basis, y_mean, y_basis, hidden_dim=8, layers=3, norm="batch", dropout=0.3)
    net.set_reference(X)
    label = torch.zeros(5, dtype=torch.long)
    net.train()
    assert torch.allclose(net(X, label), y_mean[label])
    # Coefficients = each training decoder's scores (in std units) reproduce it.
    _, basis, std = pca_basis(Y)
    coeff = ((Y - y_mean[0]) @ basis.T) / std
    assert torch.allclose(y_mean[0] + coeff @ y_basis[0], Y, atol=1e-4)


def test_only_the_mlp_is_trainable():
    X, Y = _decoders(5, 30, 5), _decoders(5, 12, 6)
    net = PCAHyperNet(*pca_basis(X)[:2], *output_bases([Y]), hidden_dim=8, layers=3)
    trainable = {id(p) for p in net.parameters()}
    assert trainable == {id(p) for p in net.hyper.parameters()}
    groups = net.param_groups(0.1)
    assert sum(len(g["params"]) for g in groups) == len(trainable)


def test_starts_at_a_given_origin_and_can_reach_the_mean():
    X, Y = _decoders(5, 30, 7), _decoders(5, 12, 8)
    floor = _decoders(1, 12, 9)
    origin, y_basis = output_bases([Y], origins=floor)
    assert torch.equal(origin, floor) and y_basis.shape == (1, 5, 12)     # mean direction + 4
    assert torch.allclose(origin[0] + y_basis[0, 0], Y.mean(0), atol=1e-5)
    net = PCAHyperNet(*pca_basis(X)[:2], origin, y_basis, hidden_dim=8, layers=3)
    net.set_reference(X)
    label = torch.zeros(5, dtype=torch.long)
    assert torch.allclose(net(X, label), floor.expand(5, -1))
