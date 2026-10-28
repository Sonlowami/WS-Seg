import torch

from training.train_loop import band_indices, stratified_indices


def test_band_is_any_channel_inside_alpha():
    alpha = 5.0
    target = torch.tensor([[5.0, 5.0], [4.9, 5.0], [5.0, -2.0], [-5.0, -5.0]])
    assert band_indices(target, alpha).tolist() == [1, 2]


def test_stratified_draws_the_requested_band_share():
    torch.manual_seed(0)
    band = torch.arange(990, 1000)
    idx = stratified_indices(1000, band, 1000, 0.5)
    assert len(idx) == 1000
    assert ((idx >= 990).sum() >= 500)                 # at least the band half
    assert idx.min() >= 0 and idx.max() < 1000


def test_zero_fraction_is_the_old_uniform_draw():
    torch.manual_seed(3)
    new = stratified_indices(1000, torch.arange(10), 64, 0.0)
    torch.manual_seed(3)
    assert torch.equal(new, torch.randint(1000, (64,)))


def test_empty_band_falls_back_to_uniform():
    torch.manual_seed(1)
    idx = stratified_indices(100, torch.empty(0, dtype=torch.long), 32, 0.5)
    assert len(idx) == 32


def test_fraction_out_of_range_is_rejected():
    try:
        stratified_indices(100, torch.arange(10), 32, 1.5)
    except ValueError:
        return
    raise AssertionError("band_fraction 1.5 was accepted")
