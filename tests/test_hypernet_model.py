import torch
import torch.nn as nn

from experiments.hypernet.model import HyperMLP, hidden_block


def _model(**kw):
    torch.manual_seed(0)
    m = HyperMLP(in_dim=6, out_dim=4, n_labels=2, hidden_dim=8, layers=3, **kw)
    m.set_reference(torch.randn(5, 6), torch.randn(2, 4), torch.rand(4) + 0.5)
    return m


def test_hidden_block_order():
    kinds = [type(m) for m in hidden_block(4, 8, "relu", "batch", 0.3)]
    assert kinds == [nn.Linear, nn.BatchNorm1d, nn.ReLU, nn.Dropout]
    assert [type(m) for m in hidden_block(4, 8)] == [nn.Linear, nn.GELU]


def test_starts_at_base_with_every_regularizer():
    for norm in (None, "batch", "layer"):
        m = _model(norm=norm, dropout=0.5)
        m.train()
        label = torch.tensor([0, 1, 1])
        assert torch.equal(m(torch.randn(3, 6), label), m.base[label])


def test_nothing_after_the_output_layer():
    # Dropout acts on the hidden activations feeding the output Linear, never
    # on the predicted weights themselves.
    m = _model(norm="batch", dropout=0.5)
    assert isinstance(m.mlp[-1], nn.Linear) and m.mlp[-1].out_features == 4
    assert sum(isinstance(x, nn.BatchNorm1d) for x in m.mlp) == 2


def test_weight_decay_only_on_linear_weights():
    m = _model(norm="batch", dropout=0.1)
    decay, rest = m.param_groups(0.1)
    linears = [x for x in m.mlp if isinstance(x, nn.Linear)]
    assert {id(p) for p in decay["params"]} == {id(x.weight) for x in linears}
    assert rest["weight_decay"] == 0.0
    assert len(decay["params"]) + len(rest["params"]) == len(list(m.parameters()))


def test_eval_mode_is_deterministic_and_leaves_batch_stats():
    m = _model(norm="batch", dropout=0.5)
    nn.init.normal_(m.mlp[-1].weight)              # make the output depend on the input
    x, label = torch.randn(4, 6), torch.tensor([0, 1, 0, 1])
    m.eval()
    stats = m.mlp[1].running_mean.clone()
    assert torch.equal(m(x, label), m(x, label))
    assert torch.equal(m.mlp[1].running_mean, stats)
