import torch

from experiments.hypernet.ridge import PCARidge


def _data(n=9, d=20, p=5, seed=0):
    g = torch.Generator().manual_seed(seed)
    X = torch.randn(n, d, generator=g, dtype=torch.float64)
    Y = X[:, :3] @ torch.randn(3, p, generator=g, dtype=torch.float64) + 0.1 * torch.randn(n, p, generator=g, dtype=torch.float64)
    groups = [f"case{i // 2}" for i in range(n)]          # pairs of rows share a case
    return X, Y, groups


def _ridge(Z, Yc, lam):
    return torch.linalg.solve(Z.T @ Z + lam * torch.eye(Z.shape[1], dtype=Z.dtype), Z.T @ Yc)


def test_closed_form_matches_explicit_ridge_on_pca_scores():
    X, Y, groups = _data()
    model = PCARidge(components=4, alphas=[0.5]).fit(X, Y, groups)
    Xc, Yc = X - X.mean(0), Y - Y.mean(0)
    Z = Xc @ model.V
    lam = 0.5 * float((torch.linalg.svdvals(Xc)[:4] ** 2).mean())
    expected = Y.mean(0) + Z @ _ridge(Z, Yc, lam)
    assert torch.allclose(model.predict(X).double(), expected, atol=1e-5)


def test_leave_one_case_out_matches_refitting():
    X, Y, groups = _data()
    model = PCARidge(components=4, alphas=[0.5]).fit(X, Y, groups)
    # Brute force with the PCA basis held fixed (as documented) and the
    # training means and ridge refitted without each case.
    Z = (X - X.mean(0)) @ model.V
    lam = 0.5 * float((torch.linalg.svdvals(X - X.mean(0))[:4] ** 2).mean())
    err = 0.0
    for case in sorted(set(groups)):
        out = torch.tensor([g == case for g in groups])
        Zi = torch.cat([torch.ones(int((~out).sum()), 1, dtype=Z.dtype), Z[~out]], 1)
        # Unpenalized intercept, penalized slopes.
        P = lam * torch.eye(Zi.shape[1], dtype=Z.dtype)
        P[0, 0] = 0
        W = torch.linalg.solve(Zi.T @ Zi + P, Zi.T @ Y[~out])
        pred = torch.cat([torch.ones(int(out.sum()), 1, dtype=Z.dtype), Z[out]], 1) @ W
        err += float(((Y[out] - pred) ** 2).sum())
    assert abs(model.search[0]["loco_mse"] - err / Y.numel()) < 1e-8


def test_components_are_capped_at_n_minus_one():
    X, Y, groups = _data(n=6)
    assert PCARidge(components=50, alphas=[1.0]).fit(X, Y, groups).k == 5
    assert PCARidge(components=None, alphas=[1.0]).fit(X, Y, groups).k == 5
