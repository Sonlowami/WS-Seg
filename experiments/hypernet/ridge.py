"""
The high-bias baseline for the hypernetwork: ridge regression on PCA
coordinates of the intensity decoders' weights.

  X (n x d) training intensity decoders, Y (n x p) the matching SDF decoders
  (direct per-case fits, the ceiling's procedure), both centred on their
  training means; PCA of X keeps k <= n - 1 components, Z = Xc V_k; ridge
  from Z to Yc in closed form. Since Z's columns are orthogonal (Z = U_k S_k),
  the ridge solution is B = diag(S / (S^2 + lam)) U_k^T Yc, and a test
  decoder x maps to y_mean + ((x - x_mean) V_k) B.

The penalty is chosen by leave-one-case-out error, in closed form: with
H = 1 1^T / n + U diag(S^2 / (S^2 + lam)) U^T, the held-out residuals of the
rows g of one case are (I - H_gg)^-1 r_g. Whole cases are left out because
a case's modalities share one Y; leaving out one row would leave its twin
in. PCA itself is not refitted per fold, so the error is slightly
optimistic. Errors are measured in units of each tensor's init scale
(HeadLayout.tensor_stds), so the large first-layer tensor does not decide
alone.
"""
import torch


class PCARidge:
    def __init__(self, components=None, alphas=(1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0)):
        """components: k, a list of k to choose from, or None for n - 1.
        alphas: penalties relative to the mean of S^2 (scale free)."""
        self.components, self.alphas = components, alphas

    def fit(self, X: torch.Tensor, Y: torch.Tensor, groups: list, y_scale: torch.Tensor = None):
        """X (n, d), Y (n, p) float tensors; groups[i] names row i's case."""
        X, Y = X.double(), Y.double()
        n = X.shape[0]
        if n < 2:
            raise ValueError(f"PCA + ridge needs at least 2 training rows, got {n}")
        self.x_mean, self.y_mean = X.mean(0), Y.mean(0)
        Xc, Yc = X - self.x_mean, Y - self.y_mean
        U, S, Vh = torch.linalg.svd(Xc, full_matrices=False)
        rank = int((S > S[0] * 1e-10).sum())
        max_k = min(n - 1, rank)
        if self.components is None:
            ks = [max_k]
        else:
            ks = sorted({min(int(k), max_k) for k in
                         (self.components if isinstance(self.components, (list, tuple)) else [self.components])})
        scale = torch.ones(Y.shape[1], dtype=Y.dtype) if y_scale is None else y_scale.double().cpu()
        rows = {}
        for i, g in enumerate(groups):
            rows.setdefault(g, []).append(i)
        folds = [torch.tensor(r) for r in rows.values()]

        # The mean decoder's own held-out error: what ridge has to beat.
        total = Yc.sum(0)
        self.mean_loco_mse = sum(float((((Yc[g] - (total - Yc[g].sum(0)) / (n - len(g))) / scale) ** 2).sum())
                                 for g in folds) / (n * Y.shape[1]) if len(folds) > 1 else float("nan")
        self.search = []
        for k in ks:
            Uk, Sk = U[:, :k], S[:k]
            UtY = Uk.T @ Yc
            for a in self.alphas:
                lam = a * float((Sk ** 2).mean())
                shrink = Sk ** 2 / (Sk ** 2 + lam)
                residual = Yc - Uk @ (shrink[:, None] * UtY)
                err = 0.0
                for g in folds:
                    H = (Uk[g] * shrink) @ Uk[g].T + 1.0 / n
                    held_out = torch.linalg.solve(torch.eye(len(g), dtype=H.dtype) - H, residual[g])
                    err += float(((held_out / scale) ** 2).sum())
                self.search.append({"k": k, "alpha": a, "loco_mse": err / (n * Y.shape[1])})
        best = min(self.search, key=lambda r: r["loco_mse"])
        self.k, self.alpha = best["k"], best["alpha"]
        Sk = S[:self.k]
        lam = self.alpha * float((Sk ** 2).mean())
        self.V = Vh[:self.k].T                                        # (d, k)
        self.B = (Sk / (Sk ** 2 + lam))[:, None] * (U[:, :self.k].T @ Yc)   # (k, p)
        self.explained = float((Sk ** 2).sum() / (S ** 2).sum())
        return self

    def predict(self, X: torch.Tensor) -> torch.Tensor:
        """(m, d) intensity decoders -> (m, p) SDF decoders."""
        X = X.double().to(self.V.device)
        return (self.y_mean + ((X - self.x_mean) @ self.V) @ self.B).float()
