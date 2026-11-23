import torch

from experiments.hypernet.model import HyperMLP
from experiments.slices.transformer import SliceTransformer, sequence_predictor

IN, OUT, LABELS = 12, 7, 2


def _items(cases=("a", "b"), lengths=(5, 3)):
    torch.manual_seed(0)
    items = []
    for case, n in zip(cases, lengths):
        for k in range(n):
            x = torch.cat([torch.randn(IN), torch.tensor([k / (n - 1)])])
            for label in range(LABELS):
                items.append({"case_id": case, "modality": "MRI", "k": k, "position": k / (n - 1),
                              "label_index": label, "x": x})
    return items


def _net(context=True, seed=0):
    torch.manual_seed(seed)
    net = SliceTransformer(IN, OUT, LABELS, d_model=16, layers=2, heads=4, dropout=0.0, context=context)
    net.set_reference(torch.randn(20, IN), torch.randn(LABELS, OUT), torch.rand(OUT) + 0.5)
    torch.nn.init.normal_(net.head.weight)
    return net.eval()


def test_parameter_count_matches_the_slice_hypernet():
    decoder = 256 * 64 + 64 + 64 + 1                            # 64-wide slice decoder head
    net = SliceTransformer(decoder, decoder, 2, 512, 6, 4)
    mlp = HyperMLP(decoder + 1, decoder, 2, 1024, 4, "gelu", "batch", 0.2)
    count = lambda m: sum(p.numel() for p in m.parameters())
    assert abs(count(net) / count(mlp) - 1) < 0.02


def test_starts_at_the_base_decoder():
    torch.manual_seed(0)
    net = SliceTransformer(IN, OUT, LABELS, d_model=16, layers=2, heads=4).eval()
    net.set_reference(torch.randn(20, IN), torch.randn(LABELS, OUT), torch.ones(OUT))
    items = _items()
    out = sequence_predictor(net, items, "cpu")(items)
    assert torch.equal(out, net.base[torch.tensor([i["label_index"] for i in items])])


def test_layers_start_with_different_weights():
    net = _net()
    a, b = (l.linear1.weight for l in net.layers)
    assert not torch.equal(a, b)


def test_padding_and_batching_do_not_change_a_prediction():
    net, items = _net(), _items()
    predict = sequence_predictor(net, items, "cpu")
    short = [i for i in items if i["case_id"] == "b"]           # 3 slices, padded to 5 when batched
    with torch.no_grad():
        alone, together = predict(short), predict(items)
    idx = [n for n, i in enumerate(items) if i["case_id"] == "b"]
    assert torch.allclose(alone, together[idx], atol=1e-5)


def test_padding_is_finite_and_harmless_without_context():
    net, items = _net(context=False), _items()
    predict = sequence_predictor(net, items, "cpu")
    short = [i for i in items if i["case_id"] == "b"]
    with torch.no_grad():
        together = predict(items)
        alone = predict(short)
    idx = [n for n, i in enumerate(items) if i["case_id"] == "b"]
    assert torch.isfinite(together).all() and torch.allclose(alone, together[idx], atol=1e-5)


def test_context_reaches_other_slices_only_with_context():
    items = _items()
    edit = lambda i: i["case_id"] == "a" and i["k"] == 0                       # one slice of volume a
    changed = [{**i, "x": i["x"] + (1.0 if edit(i) else 0.0)} for i in items]
    for context, should_change in ((True, True), (False, False)):
        net = _net(context)
        with torch.no_grad():
            a = sequence_predictor(net, items, "cpu")(items)
            b = sequence_predictor(net, changed, "cpu")(changed)
        other = [n for n, i in enumerate(items) if i["case_id"] == "a" and i["k"] != 0]
        assert (not torch.allclose(a[other], b[other], atol=1e-6)) == should_change
        case_b = [n for n, i in enumerate(items) if i["case_id"] == "b"]
        assert torch.allclose(a[case_b], b[case_b], atol=1e-6)               # never across volumes


def test_trains_with_finite_gradients():
    net, items = _net(), _items()
    net.train()
    predict = sequence_predictor(net, items, "cpu")
    opt = torch.optim.AdamW(net.param_groups(0.1), lr=1e-3)
    for _ in range(3):
        loss = (predict(items[:6]) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        assert all(torch.isfinite(p.grad).all() for p in net.parameters() if p.grad is not None)
        opt.step()
