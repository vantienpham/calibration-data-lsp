"""Ordered whitened bases: the training-free initialization of every unit.

Input side. With calibration Gram G = S S^T (Cholesky), the whitened truncation
of the row-stacked weight W of a unit keeps the leading right singular
directions of W S. Its kept input subspace is span(S V_r); the orthogonal
projector onto it removes span(S^{-T} V_{>r}), so the removed basis for k
directions is qf(S^{-T} V_{>r}) with V_{>r} the trailing k right singular
vectors (Bini et al., 2026, Prop. 1).

Output side. The bias-free output Gram W G W^T has the left singular vectors of
W S as eigenvectors; removing the eigenvectors with the k smallest eigenvalues
is the optimal whitened truncation itself.

Bases are computed in float64 and returned in float64.
"""

from __future__ import annotations

import torch

from .arch import Unit


def robust_cholesky(G: torch.Tensor, max_tries: int = 6) -> tuple[torch.Tensor, float]:
    """Lower Cholesky factor of a PSD matrix, with a ridge only if needed.

    Checks the factor itself for non-finite entries: on some GPUs cholesky_ex
    reports success alongside a NaN factor (observed on older NVIDIA GPUs).
    """
    G = 0.5 * (G + G.T)
    L, info = torch.linalg.cholesky_ex(G)
    if int(info) == 0 and bool(torch.isfinite(L).all()):
        return L, 0.0
    evals = torch.linalg.eigvalsh(G)
    scale = float(G.diagonal().mean().abs().clamp_min(1e-30))
    eta = max(-float(evals.min()), 0.0) + 1e-6 * scale
    eye = torch.eye(G.shape[0], dtype=G.dtype, device=G.device)
    for _ in range(max_tries):
        L, info = torch.linalg.cholesky_ex(G + eta * eye)
        if int(info) == 0 and bool(torch.isfinite(L).all()):
            return L, eta
        eta *= 10.0
    raise torch.linalg.LinAlgError("Cholesky failed even with a ridge")


def _svd_right(M: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Singular values (descending) and right singular vectors (columns) of M."""
    m, d = M.shape
    try:
        if m >= d:
            _, s, vh = torch.linalg.svd(M, full_matrices=False)
            return s, vh.T
        evals, evecs = torch.linalg.eigh(M.T @ M)
        order = torch.argsort(evals, descending=True)
        return evals[order].clamp_min(0).sqrt(), evecs[:, order]
    except torch.linalg.LinAlgError:
        # cuSOLVER can fail to converge where CPU LAPACK succeeds (older NVIDIA GPUs).
        s, v = _svd_right(M.cpu())
        return s.to(M.device), v.to(M.device)


class InBasis:
    """Ordered whitened basis of an input-side unit (members row-stacked)."""

    def __init__(self, weights: list[torch.Tensor], G: torch.Tensor):
        G = G.double()
        self.S, self.ridge = robust_cholesky(G)
        M = torch.cat([w.double() for w in weights], 0) @ self.S
        self.sv, self.Vw = _svd_right(M)          # Vw: d x d, columns by decreasing sv
        self.d = G.shape[0]

    def to(self, device) -> "InBasis":
        self.S, self.sv, self.Vw = self.S.to(device), self.sv.to(device), self.Vw.to(device)
        return self

    def removed(self, k: int, device=None) -> torch.Tensor:
        """Orthonormal basis (d x k) of the subspace removed by the recast truncation."""
        dev = device or self.S.device
        if k <= 0:
            return torch.zeros(self.d, 0, dtype=torch.float64, device=dev)
        S = self.S.to(dev)
        Y = torch.linalg.solve_triangular(S.T, self.Vw[:, -k:].to(dev), upper=True)
        q, _ = torch.linalg.qr(Y)
        return q

    def whitened_error(self, k: int) -> float:
        """Layer-output error of the (oblique) whitened truncation at rank d - k."""
        return float((self.sv[self.d - k:] ** 2).sum()) if k > 0 else 0.0

    def oblique_factors(self, weights: list[torch.Tensor], k: int, device=None):
        """Factors of the whitened truncation itself (SVD-LLM), not its orthogonal recast.

        W_hat = W S V_r V_r^T S^{-1} = B A with the shared A = V_r^T S^{-1} (r x d_in)
        and B_m = W_m S V_r. Same shapes, so the same parameter count, as the
        projector merge; it differs from it only in the kept input map.
        """
        dev = device or self.S.device
        S = self.S.to(dev)
        Vr = self.Vw[:, : self.d - k].to(dev)
        # A = V_r^T S^{-1}: solve X S = V_r^T for X.
        A = torch.linalg.solve_triangular(S, Vr.T, upper=False, left=False)
        Bs = [w.double().to(dev) @ (S @ Vr) for w in weights]
        return A, Bs


class OutBasis:
    """Ordered basis of an output-side unit from its bias-free output Gram."""

    def __init__(self, Gout: torch.Tensor):
        G = 0.5 * (Gout.double() + Gout.double().T)
        try:
            self.evals, self.E = torch.linalg.eigh(G)   # ascending
        except torch.linalg.LinAlgError:
            e, v = torch.linalg.eigh(G.cpu())
            self.evals, self.E = e.to(G.device), v.to(G.device)
        self.d = G.shape[0]

    def to(self, device) -> "OutBasis":
        self.evals, self.E = self.evals.to(device), self.E.to(device)
        return self

    def removed(self, k: int, device=None) -> torch.Tensor:
        dev = device or self.E.device
        if k <= 0:
            return torch.zeros(self.d, 0, dtype=torch.float64, device=dev)
        return self.E[:, :k].to(dev).contiguous()

    def whitened_error(self, k: int) -> float:
        return float(self.evals[:k].clamp_min(0).sum()) if k > 0 else 0.0


def build_bases(model, units: list[Unit], grams: dict[str, torch.Tensor],
                park: str | None = "cpu") -> dict[str, object]:
    """InBasis / OutBasis for every unit, computed where the Grams live.

    Each basis is moved to ``park`` once built (CPU by default): a d x d float64
    factor per unit adds up to tens of GB on 4096-wide models.
    """
    bases = {}
    for u in units:
        if u.side == "in":
            ws = [model.get_submodule(p).weight.detach() for p in u.members]
            b = InBasis(ws, grams[u.name])
        else:
            b = OutBasis(grams[u.name])
        bases[u.name] = b.to(park) if park else b
    return bases
