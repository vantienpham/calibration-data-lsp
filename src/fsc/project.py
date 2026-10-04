"""Projector modules, the bank that owns them, and the exact merge into low-rank factors.

A unit removes k directions with the orthogonal projector P = I - U U^T, where
U = qf(V) is the orthonormal factor of the thin QR of an unconstrained V (d x k).
Input-side units apply P to the shared input of all members, output-side units
to the (bias-free) output of their single member. The projection runs in
activation space, so no dense projected weight is ever formed.

During training the effective operator is P_{alpha,m} = I - alpha U diag(m) U^T:
alpha ramps from 0 to 1 over the first epoch and m is a Bernoulli keep-mask
(direction dropout). The mask is shared by the members of a tied group, which
lets the group project its input once.

After training, ``merged`` swaps every unit for its exact factors
W P = (W U_perp) U_perp^T on the input side, P W = U_perp (U_perp^T W) on the
output side; an input-tied group shares its A = U_perp^T.
"""

from __future__ import annotations

from contextlib import contextmanager

import torch
import torch.nn as nn
import torch.nn.functional as F

from .arch import Unit


def _key(name: str) -> str:
    return name.replace(".", "_")


def _parent(model: nn.Module, path: str) -> tuple[nn.Module, str]:
    parent_path, attr = path.rsplit(".", 1)
    return model.get_submodule(parent_path), attr


class Projector(nn.Module):
    """Removes span(qf(V)) from a d-dimensional activation."""

    def __init__(self, V: torch.Tensor):
        super().__init__()
        self.V = nn.Parameter(V.detach().to(torch.float32).clone())
        self.alpha = 1.0
        self.active = True
        self._Uc: torch.Tensor | None = None   # basis in compute dtype (graph to V while training)
        self._U_graph: torch.Tensor | None = None  # qf(V) with its graph, one per step
        self._U_leaf: torch.Tensor | None = None   # detached copy the micro-batches differentiate
        self._mask: torch.Tensor | None = None
        self._key: torch.Tensor | None = None  # identity cache: the input seen last
        self._val: torch.Tensor | None = None

    @property
    def k(self) -> int:
        return self.V.shape[1] if self._Uc is None else self._Uc.shape[1]

    def basis(self) -> torch.Tensor:
        """Orthonormal basis of span(V); differentiable through QR."""
        if self.V.shape[1] == 0:
            return self.V
        q, _ = torch.linalg.qr(self.V, mode="reduced")
        return q

    def refresh(self, dtype: torch.dtype, drop_p: float = 0.0,
                generator: torch.Generator | None = None) -> None:
        """Recompute U from V; call before each forward pass that should see V."""
        self._Uc = self.basis().to(dtype)
        if drop_p > 0 and self.k > 0:
            keep = torch.rand(self.k, device=self.V.device, generator=generator) >= drop_p
            self._mask = keep.to(dtype)
        else:
            self._mask = None
        self.clear()

    def begin_step(self, drop_p: float = 0.0, generator: torch.Generator | None = None) -> None:
        """One QR per optimizer step: micro-batches accumulate into a detached basis."""
        self._U_graph = self.basis()
        self._U_leaf = self._U_graph.detach().requires_grad_(True)
        if drop_p > 0 and self.k > 0:
            keep = torch.rand(self.k, device=self.V.device, generator=generator) >= drop_p
            self._mask = keep.float()
        else:
            self._mask = None

    def begin_micro(self, dtype: torch.dtype) -> None:
        """Cast the step's basis for one micro-batch (a fresh graph node each time)."""
        self._Uc = self._U_leaf.to(dtype)
        if self._mask is not None:
            self._mask = self._mask.to(dtype)
        self.clear()

    def end_step(self) -> tuple[torch.Tensor, torch.Tensor] | None:
        """(U with graph, accumulated dL/dU) for the single backward through QR."""
        out = None
        if self._U_leaf is not None and self._U_leaf.grad is not None:
            out = (self._U_graph, self._U_leaf.grad)
        self._U_graph = self._U_leaf = None
        return out

    def set_basis(self, U: torch.Tensor, dtype: torch.dtype) -> None:
        """Use a fixed orthonormal basis (no gradient); for allocation measurements."""
        self._Uc = U.to(dtype)
        self._mask = None
        self.clear()

    def clear(self) -> None:
        self._key = None
        self._val = None

    def idle(self) -> bool:
        return (not self.active) or self.alpha == 0.0 or self._Uc is None or self.k == 0

    def remove(self, x: torch.Tensor) -> torch.Tensor:
        U = self._Uc
        c = x @ U
        if self._mask is not None:
            c = c * self._mask
        if self.alpha != 1.0:
            c = c * self.alpha
        return x - c @ U.T

    def project_shared(self, x: torch.Tensor) -> torch.Tensor:
        """Project an input shared by several members, computing it once per forward."""
        if self.idle():
            return x
        if x is self._key:
            return self._val
        y = self.remove(x)
        self._key, self._val = x, y
        return y


class InProj(nn.Module):
    """Member of an input-side unit: W (P x) + b."""

    def __init__(self, linear: nn.Linear, proj: Projector):
        super().__init__()
        self.linear = linear
        object.__setattr__(self, "proj", proj)   # owned by the bank, not by the model

    def forward(self, x):
        return self.linear(self.proj.project_shared(x))


class OutProj(nn.Module):
    """Output-side unit: P (W x) + b, the bias left unchanged."""

    def __init__(self, linear: nn.Linear, proj: Projector):
        super().__init__()
        self.linear = linear
        object.__setattr__(self, "proj", proj)

    def forward(self, x):
        y = self.linear(x)
        p = self.proj
        if p.idle():
            return y
        b = self.linear.bias
        return p.remove(y) if b is None else p.remove(y - b) + b


class ProjectorBank(nn.Module):
    """All projectors of a model. Holds the only trainable parameters (the V's)."""

    def __init__(self, units: list[Unit], Vs: dict[str, torch.Tensor]):
        super().__init__()
        self.units = [u for u in units if u.name in Vs]
        self.projs = nn.ModuleDict({_key(u.name): Projector(Vs[u.name]) for u in self.units})
        self._swaps: list[tuple[nn.Module, str, nn.Module]] = []

    def __getitem__(self, name: str) -> Projector:
        return self.projs[_key(name)]

    # -- wiring ---------------------------------------------------------------
    def attach(self, model: nn.Module) -> None:
        assert not self._swaps, "bank already attached"
        for u in self.units:
            p = self[u.name]
            for path in u.members:
                parent, attr = _parent(model, path)
                lin = getattr(parent, attr)
                wrapper = InProj(lin, p) if u.side == "in" else OutProj(lin, p)
                setattr(parent, attr, wrapper)
                self._swaps.append((parent, attr, lin))

    def detach(self, model: nn.Module | None = None) -> None:
        for parent, attr, lin in reversed(self._swaps):
            setattr(parent, attr, lin)
        self._swaps = []

    # -- state ----------------------------------------------------------------
    def set_active(self, flag: bool) -> None:
        for p in self.projs.values():
            p.active = flag

    def set_alpha(self, alpha: float) -> None:
        for p in self.projs.values():
            p.alpha = float(alpha)

    def refresh(self, dtype, drop_p=0.0, generator=None) -> None:
        for p in self.projs.values():
            p.refresh(dtype, drop_p, generator)

    def clear(self) -> None:
        for p in self.projs.values():
            p.clear()

    def begin_step(self, drop_p=0.0, generator=None) -> None:
        for p in self.projs.values():
            p.begin_step(drop_p, generator)

    def begin_micro(self, dtype) -> None:
        for p in self.projs.values():
            p.begin_micro(dtype)

    def end_step(self) -> None:
        """Backpropagate the step's accumulated basis gradients through QR, once."""
        outs, grads = [], []
        for p in self.projs.values():
            r = p.end_step()
            if r is not None:
                outs.append(r[0])
                grads.append(r[1])
        if outs:
            torch.autograd.backward(outs, grads)

    def ortho_penalty(self) -> torch.Tensor:
        """Mean absolute off-diagonal inner product between the columns of each V."""
        total = None
        for p in self.projs.values():
            k = p.V.shape[1]
            if k < 2:
                continue
            g = p.V.T @ p.V
            off = (g.abs().sum() - g.diagonal().abs().sum()) / (k * (k - 1))
            total = off if total is None else total + off
        if total is None:
            return torch.zeros((), device=next(self.parameters()).device)
        return total

    def export(self) -> dict[str, torch.Tensor]:
        """Orthonormal removed bases U = qf(V), on CPU, float32."""
        with torch.no_grad():
            return {u.name: self[u.name].basis().detach().float().cpu() for u in self.units}

    def raw(self) -> dict[str, torch.Tensor]:
        return {u.name: self[u.name].V.detach().float().cpu().clone() for u in self.units}

    def load_raw(self, Vs: dict[str, torch.Tensor]) -> None:
        with torch.no_grad():
            for u in self.units:
                self[u.name].V.copy_(Vs[u.name].to(self[u.name].V.device))


# ----------------------------------------------------------------------------
# Merging
# ----------------------------------------------------------------------------

def complement_basis(U: torch.Tensor) -> torch.Tensor:
    """Orthonormal basis of span(U)^perp, for U (d x k) with orthonormal columns."""
    d, k = U.shape
    q, _ = torch.linalg.qr(U.double(), mode="complete")
    return q[:, k:]


def orthonormalize(V: torch.Tensor) -> torch.Tensor:
    q, _ = torch.linalg.qr(V.double(), mode="reduced")
    return q


class LowRankLinear(nn.Module):
    """y = B (A x) + b. A may be the same Parameter object across a tied group."""

    def __init__(self, A: nn.Parameter, B: torch.Tensor, bias: torch.Tensor | None):
        super().__init__()
        self.A = A
        self.B = nn.Parameter(B, requires_grad=False)
        self.bias = None if bias is None else nn.Parameter(bias.detach().clone(), requires_grad=False)
        self.in_features = A.shape[1]
        self.out_features = B.shape[0]

    def forward(self, x):
        return F.linear(F.linear(x, self.A), self.B, self.bias)


def merge_factors(model: nn.Module, unit: Unit, U: torch.Tensor, dtype: torch.dtype | None = None):
    """Exact factors of a unit whose removed subspace has orthonormal basis U.

    Returns (A_shared_or_None, [(path, A, B, bias), ...]) on the weight's device,
    computed in float64 and returned in ``dtype`` if given (float64 otherwise).
    Casting here, one unit at a time, keeps the float64 intermediates of a single
    unit alive instead of those of the whole model (39 GB for Llama-3.1-8B at -30%).
    Input side: A = U_perp^T (shared), B_m = W_m U_perp. Output side: B = U_perp,
    A = U_perp^T W.
    """
    cast = (lambda t: t.to(dtype)) if dtype is not None else (lambda t: t)
    out = []
    lin0 = model.get_submodule(unit.members[0])
    lin0 = getattr(lin0, "linear", lin0)
    dev = lin0.weight.device
    Up = complement_basis(U.to(dev))
    if unit.side == "in":
        A = cast(Up.T.contiguous())          # one tensor object, shared by the group
        for path in unit.members:
            lin = model.get_submodule(path)
            lin = getattr(lin, "linear", lin)
            out.append((path, A, cast(lin.weight.double() @ Up), lin.bias))
        return A, out
    path = unit.members[0]
    lin = model.get_submodule(path)
    lin = getattr(lin, "linear", lin)
    out.append((path, cast(Up.T @ lin.weight.double()), cast(Up.contiguous()), lin.bias))
    return None, out


@contextmanager
def swapped(model: nn.Module, parts: list[tuple[str, torch.Tensor, torch.Tensor, object]],
            dtype: torch.dtype | None = None):
    """Replace members by LowRankLinear(A, B, bias) for every (path, A, B, bias) in parts.

    Parts that pass the same A tensor object share one stored A Parameter (tied
    groups). The original modules are restored on exit.
    """
    swaps, shared = [], {}
    try:
        for path, A, B, bias in parts:
            parent, attr = _parent(model, path)
            lin = getattr(parent, attr)
            dt = dtype or getattr(lin, "linear", lin).weight.dtype
            if id(A) not in shared:
                shared[id(A)] = nn.Parameter(A.to(dt), requires_grad=False)
            setattr(parent, attr, LowRankLinear(shared[id(A)], B.to(dt), bias))
            swaps.append((parent, attr, lin))
        yield model
    finally:
        for parent, attr, lin in reversed(swaps):
            setattr(parent, attr, lin)


@contextmanager
def merged(model: nn.Module, units: list[Unit], bases: dict[str, torch.Tensor],
           dtype: torch.dtype | None = None):
    """Swap every unit with a non-empty removed basis for its exact low-rank factors.

    ``bases`` maps unit name -> U (d x k, orthonormal columns). The original
    linear layers are restored on exit, so the same model object can be trained
    afterwards.
    """
    parts = []
    for u in units:
        U = bases.get(u.name)
        if U is None or U.shape[1] == 0:
            continue
        _, p = merge_factors(model, u, U, dtype)
        parts += p
    with swapped(model, parts, dtype):
        yield model


def stored_params(model: nn.Module, units: list[Unit]) -> int:
    """Weights currently stored by the members of ``units`` (shared tensors once)."""
    seen, total = set(), 0
    for u in units:
        for path in u.members:
            mod = model.get_submodule(path)
            for name, p in mod.named_parameters():
                if name.endswith("bias"):
                    continue
                if id(p) in seen:
                    continue
                seen.add(id(p))
                total += p.numel()
    return total


# ----------------------------------------------------------------------------
# Free low-rank factors (the unconstrained control)
# ----------------------------------------------------------------------------

class FactorLinear(nn.Module):
    """Trainable factors B (A x) + b, with the frozen dense layer as the teacher path."""

    def __init__(self, linear: nn.Linear, A: nn.Parameter, B: nn.Parameter, bank: "FactorBank"):
        super().__init__()
        self.linear = linear
        object.__setattr__(self, "A", A)
        object.__setattr__(self, "B", B)
        object.__setattr__(self, "bank", bank)

    def forward(self, x):
        if not self.bank.active:
            return self.linear(x)
        dt = x.dtype
        y = F.linear(F.linear(x, self.A.to(dt)), self.B.to(dt), self.linear.bias)
        a = self.bank.alpha
        if a >= 1.0:
            return y
        # Warm-up blend (1 - a) W x + a B A x: at initialization B A = W P, so
        # this is exactly W P_alpha x, the projector's own alpha ramp.
        return (1.0 - a) * self.linear(x) + a * y


class FactorBank(nn.Module):
    """Free factors initialized at the exact merge of the given removed bases.

    At initialization the student computes exactly the projected (NoLSP) function;
    training then moves A and B freely, i.e. it can leave the pretrained weights'
    row and column spaces, which the projector parameterization cannot.
    """

    def __init__(self, model: nn.Module, units: list[Unit], bases: dict[str, torch.Tensor]):
        super().__init__()
        self.active = True
        self.alpha = 1.0
        self.units = [u for u in units if u.name in bases and bases[u.name].shape[1] > 0]
        self.A = nn.ParameterDict()
        self.B = nn.ParameterDict()
        self._members: list[tuple[str, str, str]] = []   # (path, A key, B key)
        self._swaps = []
        for u in self.units:
            A_shared, parts = merge_factors(model, u, bases[u.name])
            if A_shared is not None:
                ka = _key(u.name) + "__A"
                self.A[ka] = nn.Parameter(A_shared.float())
            for j, (path, A, B, _bias) in enumerate(parts):
                if A_shared is None:
                    ka = _key(u.name) + f"__A{j}"
                    self.A[ka] = nn.Parameter(A.float())
                kb = _key(u.name) + f"__B{j}"
                self.B[kb] = nn.Parameter(B.float())
                self._members.append((path, ka, kb))

    def attach(self, model: nn.Module) -> None:
        assert not self._swaps
        for path, ka, kb in self._members:
            parent, attr = _parent(model, path)
            lin = getattr(parent, attr)
            setattr(parent, attr, FactorLinear(lin, self.A[ka], self.B[kb], self))
            self._swaps.append((parent, attr, lin))

    def detach(self, model: nn.Module | None = None) -> None:
        for parent, attr, lin in reversed(self._swaps):
            setattr(parent, attr, lin)
        self._swaps = []

    def set_active(self, flag: bool) -> None:
        self.active = flag

    @contextmanager
    def exported(self, model: nn.Module, dtype: torch.dtype):
        """Swap members for plain LowRankLinear modules holding the trained factors."""
        self.detach(model)
        swaps, shared = [], {}
        try:
            for path, ka, kb in self._members:
                parent, attr = _parent(model, path)
                lin = getattr(parent, attr)
                if ka not in shared:
                    shared[ka] = nn.Parameter(self.A[ka].detach().to(dtype), requires_grad=False)
                setattr(parent, attr, LowRankLinear(shared[ka], self.B[kb].detach().to(dtype), lin.bias))
                swaps.append((parent, attr, lin))
            yield model
        finally:
            for parent, attr, lin in reversed(swaps):
                setattr(parent, attr, lin)
            self.attach(model)

    def raw(self) -> dict[str, torch.Tensor]:
        out = {k: v.detach().float().cpu().clone() for k, v in self.A.items()}
        out.update({k: v.detach().float().cpu().clone() for k, v in self.B.items()})
        return out

    def load_raw(self, state: dict[str, torch.Tensor]) -> None:
        with torch.no_grad():
            for k, v in self.A.items():
                v.copy_(state[k].to(v.device))
            for k, v in self.B.items():
                v.copy_(state[k].to(v.device))
