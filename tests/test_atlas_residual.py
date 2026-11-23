import torch

from experiments.hypernet.model import HyperMLP
from experiments.slices.model import AtlasResidual

ARGS = (9, 6, 2, 16, 3, "relu", "batch", 0.1)


def _net(seed=0, **kw):
    torch.manual_seed(seed)
    net = AtlasResidual(*ARGS, **kw)
    x = torch.randn(32, 9)
    net.set_reference(x, torch.randn(2, 6), torch.rand(6) + 0.5)
    return net, x, torch.randint(0, 2, (32,))


def test_starts_at_the_base_decoder():
    net, x, label = _net()
    assert torch.equal(net.eval()(x, label), net.base[label])


def test_image_trunk_starts_as_the_hypernets_hidden_layers():
    torch.manual_seed(0)
    hyper = HyperMLP(*ARGS)
    net, _, _ = _net()
    for a, b in zip(hyper.mlp[:-1].parameters(), net.image.parameters()):
        assert torch.equal(a, b)
    assert net.head.weight.shape == hyper.mlp[-1].weight.shape


def test_atlas_sees_only_the_position():
    net, x, label = _net(image_dropout=0.0)
    torch.nn.init.normal_(net.head.weight)
    net.eval()
    h = lambda x: net.atlas(torch.cat([((x - net.in_mean) / net.in_std)[:, -1:],
                                        torch.nn.functional.one_hot(label, 2).float()], -1))
    changed = x.clone()
    changed[:, :-1] += 1.0                                   # the image part only
    assert torch.equal(h(x), h(changed)) and not torch.equal(net(x, label), net(changed, label))


def test_image_dropout_drops_whole_items_in_training_only():
    net, x, label = _net(image_dropout=0.5)
    torch.nn.init.normal_(net.head.weight)
    torch.manual_seed(1)
    net.train()
    for m in net.modules():                                  # isolate the branch dropout
        if isinstance(m, (torch.nn.Dropout, torch.nn.BatchNorm1d)):
            m.eval()
    train_out = net(x, label)
    net.eval()
    with torch.no_grad():
        z = (x - net.in_mean) / net.in_std
        onehot = torch.nn.functional.one_hot(label, 2).float()
        atlas_only = net.base[label] + net.scale * net.head(net.atlas(torch.cat([z[:, -1:], onehot], -1)))
    dropped = torch.isclose(train_out, atlas_only).all(1)
    assert 0 < dropped.sum() < len(x)                        # some items atlas-only, some not


def test_trains_with_finite_gradients():
    net, x, label = _net()
    opt = torch.optim.AdamW(net.param_groups(0.1), lr=1e-2)
    for _ in range(3):
        loss = (net.train()(x, label) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        assert all(torch.isfinite(p.grad).all() for p in net.parameters() if p.grad is not None)
        opt.step()
    assert net.head.weight.abs().sum() > 0
