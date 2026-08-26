"""Physical constraints for guided video inpainting, enforced by a Krylov projection.

Motivation
----------
The masked sampler in `inpaint.pf_ode_inpaint` already enforces one linear
constraint every step -- ``x = observed`` on the known pixels -- by an explicit
blend. This module generalises that to *arbitrary linear physical constraints*
``C x = d`` (temporal flow-consistency, intensity/mass conservation, ...) and
enforces them by projecting the sampler state onto the affine set

    x  <-  x - w * Cᵀ (C Cᵀ)⁻¹ (C x - d),        0 < w <= 1

where ``w`` is a guidance strength (``w = 1`` is the exact projection). The
normal-equations system ``(C Cᵀ) y = C x - d`` is SPD and is solved **matrix
free** by conjugate gradients: only the actions ``C v`` and ``Cᵀ v`` are needed,
never the matrix. Because consecutive frames' constraint operators are nearly
identical, CG is **warm-started from the previous step / previous frame's dual**
``y`` -- reusing the Krylov iterate is what makes the control cheap frame by
frame, and the iteration count is reported so the saving is measurable.

Adjoints for free
-----------------
Every constraint here is linear, so its adjoint ``Cᵀ`` is exactly the
vector-Jacobian product of ``C``. Rather than hand-code the adjoint of
`grid_sample` (an error-prone scatter), each constraint implements only the
forward map ``apply(x)`` and the adjoint is obtained by autograd. This keeps
``Cᵀ`` provably consistent with ``C``: a projection built on a mismatched
adjoint is silently non-orthogonal and drifts.

State convention
----------------
Constraints operate on a single clip ``x`` of shape ``(T, C, H, W)`` (the
sampler squeezes its batch dimension, which is 1 for the per-instance regime).
Masks follow the sampler convention: ``1`` = unknown/generated, ``0`` = known.
"""

import numpy as np
import torch

from .flow import warp_frame


# --------------------------------------------------------------------------
# Constraints. Each defines apply(x) -> residual-space tensor and target(x).
# --------------------------------------------------------------------------

class LinearConstraint:
    """Base class. Subclasses implement `apply` (a linear map of x) and `target`."""

    name = "constraint"

    def apply(self, x):
        """Linear map C x. Returns a tensor in residual space."""
        raise NotImplementedError

    def target(self, x):
        """The right-hand side d, a constant tensor shaped like `apply(x)`."""
        raise NotImplementedError


class KnownRegion(LinearConstraint):
    """C x = observed on known pixels. The existing hard projection, as a constraint.

    Provided mainly so the framework can be validated against the sampler's
    explicit known-region blend (they must agree). The sampler keeps that explicit
    blend as the authoritative, bit-exact enforcement of observed pixels; this is
    the same operator expressed for the projection machinery.
    """

    name = "known_region"

    def __init__(self, known_mask, reference):
        # known_mask: (T,1,H,W) or broadcastable, 1 where known.
        self.known = torch.as_tensor(known_mask)
        self.reference = torch.as_tensor(reference)

    def apply(self, x):
        return self.known.to(x) * x

    def target(self, x):
        return (self.known.to(x) * self.reference.to(x))


class FlowConsistency(LinearConstraint):
    """Temporal brightness constancy along optical flow, on the generated region.

    For each pair (t-1, t), the current frame should equal the motion-compensated
    previous frame where the flow is valid:

        x[t]  ≈  warp(x[t-1], flow_{t->t-1}).

    `warp_frame` backward-warps, so `warp_frame(x[t-1], flows[t-1])` estimates
    x[t] under the `clip_flows` convention (flows[k] maps frame k+1 into k). The
    residual is restricted to pixels that are (a) unknown -- so the constraint
    guides the fill rather than fighting observed data -- and (b) flow-valid, via
    the forward-backward mask, so disoccluded pixels with no correspondence are
    dropped instead of pulled toward nonsense. Target is zero.

    This is the constraint that couples frames, and therefore the one whose
    projection operator changes slowly enough between frames for the Krylov
    warm start to pay.
    """

    name = "flow_consistency"

    def __init__(self, flows, unknown_mask, valid_masks=None):
        # flows: list length T-1, each (2,H,W); flows[k] maps frame k+1 into k.
        self.flows = [torch.as_tensor(f) for f in flows]
        # unknown_mask: (T,1,H,W), 1 = generated. Constrain only generated pixels.
        self.unknown = torch.as_tensor(unknown_mask)
        self.valid = None
        if valid_masks is not None:
            self.valid = [torch.as_tensor(m) for m in valid_masks]

    def _sel(self, t, x):
        # Selection weight for the residual at frame t: unknown AND flow-valid.
        s = self.unknown[t].to(x)  # (1,H,W)
        if self.valid is not None:
            s = s * self.valid[t - 1].to(x)[None]  # valid_masks indexed by pair
        return s

    def apply(self, x):
        T = x.shape[0]
        res = []
        for t in range(1, T):
            pred = warp_frame(x[t - 1], self.flows[t - 1].to(x))
            res.append(self._sel(t, x) * (x[t] - pred))
        return torch.stack(res)  # (T-1, C, H, W)

    def target(self, x):
        T = x.shape[0]
        return torch.zeros((T - 1,) + tuple(x.shape[1:]), device=x.device)


class IntensityConservation(LinearConstraint):
    """Total intensity per channel conserved frame to frame (a mass control).

    A global, cheap linear constraint: sum_p x[t] - sum_p x[t-1] = 0 per channel.
    Included to show the framework handles constraints that are not pixel-local;
    its `C Cᵀ` is tiny ((T-1)*C), so it is essentially free and mainly a
    demonstration that heterogeneous constraints compose.
    """

    name = "intensity_conservation"

    def apply(self, x):
        # x: (T,C,H,W) -> per-frame per-channel spatial sum, then successive diffs.
        sums = x.sum(dim=(2, 3))          # (T, C)
        return sums[1:] - sums[:-1]        # (T-1, C)

    def target(self, x):
        return torch.zeros((x.shape[0] - 1, x.shape[1]), device=x.device)


# --------------------------------------------------------------------------
# Matrix-free projection
# --------------------------------------------------------------------------

def _adjoint(constraint, x_ref, v):
    """Cᵀ v for a linear constraint, via autograd VJP (exact since C is linear)."""
    xv = x_ref.detach().requires_grad_(True)
    r = constraint.apply(xv)
    (g,) = torch.autograd.grad(r, xv, grad_outputs=v, retain_graph=False)
    return g


def _flatten(tensors):
    return torch.cat([t.reshape(-1) for t in tensors])


def _unflatten(flat, shapes):
    out, i = [], 0
    for shp in shapes:
        n = int(np.prod(shp)) if len(shp) else 1
        out.append(flat[i:i + n].reshape(shp))
        i += n
    return out


def _cg(matvec, b, x0=None, tol=1e-4, maxiter=50):
    """Conjugate gradients for an SPD matrix-free operator. Returns (x, iters, rel)."""
    x = torch.zeros_like(b) if x0 is None else x0.clone()
    r = b - matvec(x)
    p = r.clone()
    rs = torch.dot(r, r)
    bnorm = torch.sqrt(torch.dot(b, b)).clamp_min(1e-12)
    it = 0
    if torch.sqrt(rs) / bnorm < tol:
        return x, 0, float((torch.sqrt(rs) / bnorm).item())
    for it in range(1, maxiter + 1):
        Ap = matvec(p)
        denom = torch.dot(p, Ap).clamp_min(1e-30)
        alpha = rs / denom
        x = x + alpha * p
        r = r - alpha * Ap
        rs_new = torch.dot(r, r)
        rel = torch.sqrt(rs_new) / bnorm
        if rel < tol:
            break
        p = r + (rs_new / rs) * p
        rs = rs_new
    return x, it, float(rel.item())


def project(x, constraints, weight=1.0, cg_tol=1e-4, cg_maxiter=50, warm=None,
            free_mask=None, ridge=0.0):
    """Project a clip x (T,C,H,W) onto {C x = d}, as a constrained control step.

    Each sampler step chooses a minimum-effort control (correction) ``u`` that
    steers the state onto the physical constraints:

        min_u  1/2 uᵀ W u   s.t.  C(x+u) = d      (hard control)

    whose KKT system, after eliminating ``u = -W⁻¹Cᵀ y``, is the reduced normal
    system ``(C W⁻¹ Cᵀ) y = C x - d`` with ``u = -W⁻¹Cᵀ y``. Here ``free_mask``
    *is* the control metric: ``W⁻¹ = diag(free_mask)`` (zero mobility, i.e.
    infinite control cost, on observed pixels), so the update never moves data.

    `ridge` >= 0 is the **regularised (soft) control** augmentation. It solves

        min_u  1/2 uᵀ W u + 1/(2 ridge) ‖C(x+u) - d‖²

    giving ``(C W⁻¹ Cᵀ + ridge·I) y = C x - d``. It (i) handles *inconsistent*
    constraints (noisy block-match flow, disocclusion) gracefully, (ii) improves
    conditioning so warm-started CG converges faster, and (iii) recovers the exact
    projection as ``ridge -> 0``. `ridge=0` reproduces the hard projection
    bit-for-bit.

    Solved matrix-free by CG (warm-started from `warm`); returns ``x - weight·u``
    and an info dict carrying the dual `y` for the next warm start and the CG
    iteration count. `weight` < 1 damps the applied control (a softer step); it is
    orthogonal to `ridge`, which reshapes the solve itself.

    free_mask : optional (T,1,H,W) (broadcastable) with 1 on pixels the control is
        allowed to move -- the metric W⁻¹ above. Keeps observed pixels bit-exact
        and confines coupling constraints to the generated region; `apply(x)` still
        sees the full clip, so observed content informs the fill.
    """
    if not constraints:
        return x, {"cg_iters": 0, "cg_rel": 0.0, "dual": warm}

    fm = None if free_mask is None else torch.as_tensor(free_mask).to(x)

    def ctv(yblocks):
        """W⁻¹ Cᵀ y in x-space (free_mask is the metric W⁻¹)."""
        xadj = None
        for c, yb in zip(constraints, yblocks):
            g = _adjoint(c, x, yb)
            xadj = g if xadj is None else xadj + g
        return xadj if fm is None else fm * xadj

    # Residual r0 = C x - d, as a flat block vector; remember block shapes to split.
    residuals = [c.apply(x) - c.target(x) for c in constraints]
    shapes = [tuple(r.shape) for r in residuals]
    b = _flatten(residuals)

    def matvec(yflat):
        yblocks = _unflatten(yflat, shapes)
        xadj = ctv(yblocks)                       # W⁻¹ Cᵀ y
        base = _flatten([c.apply(xadj) for c in constraints])   # C W⁻¹ Cᵀ y
        return base + ridge * yflat if ridge else base          # + Tikhonov term

    x0 = warm if (warm is not None and warm.shape == b.shape) else None
    y, iters, rel = _cg(matvec, b, x0=x0, tol=cg_tol, maxiter=cg_maxiter)

    u = ctv(_unflatten(y, shapes))                # control W⁻¹ Cᵀ y
    x_new = x - weight * u
    return x_new, {"cg_iters": int(iters), "cg_rel": float(rel), "dual": y.detach(),
                   "control_norm": float(u.norm().item())}


# --------------------------------------------------------------------------
# Config -> constraint objects
# --------------------------------------------------------------------------

def build_constraints(spec, reference, unknown_mask, flows=None, valid_masks=None):
    """Build a constraint list from a config `sample.constraints` spec.

    `spec` is a list of names or {name: ...} dicts. `reference` is (T,C,H,W),
    `unknown_mask` is (T,1,H,W). `flows`/`valid_masks` are required by
    flow_consistency and are computed by the caller on the observed clip.
    """
    if not spec:
        return []
    out = []
    for item in spec:
        name = item if isinstance(item, str) else item.get("name")
        if name == "flow_consistency":
            if flows is None:
                raise ValueError("flow_consistency needs flows; none supplied")
            out.append(FlowConsistency(flows, unknown_mask, valid_masks))
        elif name == "intensity_conservation":
            out.append(IntensityConservation())
        elif name == "known_region":
            out.append(KnownRegion(1.0 - np.asarray(unknown_mask), reference))
        else:
            raise ValueError(f"unknown constraint {name!r}")
    return out


# --------------------------------------------------------------------------
# Tests (no trained network needed)
# --------------------------------------------------------------------------

def tests_constraints(verbose=True):
    """Properties every part of the projection must satisfy.

    1. Autograd adjoints are exact: <C x, v> == <x, Cᵀ v>. A projection built on
       an inexact adjoint is silently non-orthogonal.
    2. The KnownRegion projection reproduces the sampler's explicit blend.
    3. FlowConsistency drives the masked flow residual to zero on a clip with known
       ground-truth flow, while `free_mask` leaves observed pixels bit-exact.
    """
    torch.manual_seed(0)
    T, Ch, H, W = 5, 3, 12, 12
    x = torch.randn(T, Ch, H, W, dtype=torch.float64)
    unknown = torch.zeros(T, 1, H, W, dtype=torch.float64)
    unknown[:, :, 3:9, 3:9] = 1.0
    flows = [torch.zeros(2, H, W, dtype=torch.float64) for _ in range(T - 1)]
    for f in flows:
        f[0] = -1.0

    res = {}
    for c in (KnownRegion(1.0 - unknown, x.clone()),
              FlowConsistency(flows, unknown),
              IntensityConservation()):
        Cx = c.apply(x)
        v = torch.randn_like(Cx)
        lhs = float((Cx * v).sum())
        rhs = float((x * _adjoint(c, x, v)).sum())
        res[f"adjoint_{c.name}"] = abs(lhs - rhs) / (abs(lhs) + abs(rhs) + 1e-12)

    known = 1.0 - unknown
    ref = torch.randn(T, Ch, H, W, dtype=torch.float64)
    xf = torch.randn(T, Ch, H, W, dtype=torch.float64)
    xp, _ = project(xf, [KnownRegion(known, ref)], weight=1.0, cg_tol=1e-12)
    res["known_matches_blend"] = float((xp - (known * ref + unknown * xf)).abs().max())

    base = torch.randn(Ch, H, W, dtype=torch.float64)
    clip = torch.stack([torch.roll(base, shifts=t, dims=2) for t in range(T)])
    corrupt = clip + unknown * torch.randn_like(clip) * 0.8
    fc = [FlowConsistency(flows, unknown)]
    fixed, _ = project(corrupt, fc, weight=1.0, cg_tol=1e-8, cg_maxiter=300,
                       free_mask=unknown)

    def resid(cl):
        r = 0.0
        for t in range(1, T):
            d = unknown[t] * (cl[t] - warp_frame(cl[t - 1], flows[t - 1]))
            r += float((d ** 2).sum())
        return r

    res["flow_resid_before"] = resid(corrupt)
    res["flow_resid_after"] = resid(fixed)
    res["known_drift"] = float((known * (fixed - corrupt)).abs().max())

    # Augmentation: regularised (soft) control. ridge=0 must reproduce the hard
    # projection bit-for-bit; ridge>0 trades constraint fidelity for smaller
    # control effort and better conditioning (fewer CG iters).
    hard_x, hard = project(corrupt, fc, cg_tol=1e-10, cg_maxiter=400, free_mask=unknown,
                           ridge=0.0)
    hard_x2, _ = project(corrupt, fc, cg_tol=1e-10, cg_maxiter=400, free_mask=unknown)
    soft_x, soft = project(corrupt, fc, cg_tol=1e-10, cg_maxiter=400, free_mask=unknown,
                           ridge=0.5)
    res["ridge0_matches_default"] = float((hard_x - hard_x2).abs().max())
    res["ctrl_norm_hard"] = hard["control_norm"]
    res["ctrl_norm_soft"] = soft["control_norm"]
    res["cg_iters_hard"] = float(hard["cg_iters"])
    res["cg_iters_soft"] = float(soft["cg_iters"])

    if verbose:
        for k, v in res.items():
            print(f"  {k:<22} {v:.3e}")

    tol_adj = 1e-5  # warp grid is float32; other adjoints are exact
    for k, v in res.items():
        if k.startswith("adjoint_") and v > tol_adj:
            raise AssertionError(f"{k} adjoint mismatch {v:.2e}")
    if res["known_matches_blend"] > 1e-10:
        raise AssertionError("KnownRegion projection disagrees with the blend")
    if res["flow_resid_after"] > res["flow_resid_before"] * 1e-2:
        raise AssertionError("flow-consistency projection did not reduce the residual")
    if res["known_drift"] != 0.0:
        raise AssertionError("free_mask did not keep observed pixels bit-exact")
    if res["ridge0_matches_default"] != 0.0:
        raise AssertionError("ridge=0 does not reproduce the hard projection exactly")
    if not (res["ctrl_norm_soft"] < res["ctrl_norm_hard"]):
        raise AssertionError("regularised control did not reduce control effort")
    if res["cg_iters_soft"] > res["cg_iters_hard"]:
        raise AssertionError("regularised control did not improve CG conditioning")
    return res

