"""Centered orthonormal SENSE operators and batched conjugate gradient."""

import math

import torch


def fft2c(x):
    return torch.fft.fftshift(
        torch.fft.fftn(
            torch.fft.ifftshift(x, dim=(-2, -1)), dim=(-2, -1), norm="ortho"
        ),
        dim=(-2, -1),
    )


def ifft2c(x):
    return torch.fft.fftshift(
        torch.fft.ifftn(
            torch.fft.ifftshift(x, dim=(-2, -1)), dim=(-2, -1), norm="ortho"
        ),
        dim=(-2, -1),
    )


def sense_forward(x, mps, mask):
    """Apply M F S: x (B,H,W), maps (B,C,H,W), bool mask (W,)."""
    return fft2c(mps * x[:, None]) * mask


def sense_adjoint(y, mps, mask):
    """Apply S^H F^H M: y/maps (B,C,H,W), bool mask (W,)."""
    return (mps.conj() * ifft2c(y * mask)).sum(dim=1)


def _dot(a, b):
    return (a.conj() * b).real.sum(dim=(-2, -1), keepdim=True)


def _validate(ksp, mps, mask, lam, iters, tol, x0):
    if ksp.ndim != 4 or min(ksp.shape) < 1 or mps.shape != ksp.shape:
        raise ValueError(
            "k-space and maps must have matching nonempty (B,C,H,W) shapes"
        )
    if ksp.dtype not in (torch.complex64, torch.complex128) or mps.dtype != ksp.dtype:
        raise ValueError("k-space and maps must share a complex floating dtype")
    if mps.device != ksp.device or mask.device != ksp.device:
        raise ValueError("k-space, maps, and mask must share a device")
    if mask.dtype != torch.bool or mask.shape != (ksp.shape[-1],) or not mask.any():
        raise ValueError("mask must be a nonempty bool vector over phase columns")
    if not math.isfinite(lam) or lam < 0 or not math.isfinite(tol) or tol <= 0:
        raise ValueError("lambda must be nonnegative and tolerance positive and finite")
    if isinstance(iters, bool) or not isinstance(iters, int) or iters < 1:
        raise ValueError("iters must be a positive integer")
    if not torch.isfinite(ksp).all() or not torch.isfinite(mps).all():
        raise ValueError("k-space and maps must be finite")
    if x0 is not None:
        shape = (ksp.shape[0], *ksp.shape[-2:])
        if x0.shape != shape or x0.dtype != ksp.dtype or x0.device != ksp.device:
            raise ValueError(
                "initial image must match batch, geometry, dtype, and device"
            )
        if not torch.isfinite(x0).all():
            raise ValueError("initial image must be finite")


@torch.no_grad()
def cg_sense(ksp, mps, mask, lam=0.0, iters=40, tol=1e-5, x0=None):
    """Solve (A^H A + lam I)x = A^H y, independently for each batch slice.

    Returns image (B,H,W), true relative normal-equation residual (B,), and
    stopping iteration (B,). Initially solved slices report zero iterations;
    unconverged slices report ``iters``. Inspect residuals to check convergence.
    Zero right-hand sides return a zero image, including with an initial guess.
    """
    _validate(ksp, mps, mask, lam, iters, tol, x0)
    b = sense_adjoint(ksp, mps, mask)

    def normal(x):
        return sense_adjoint(sense_forward(x, mps, mask), mps, mask) + lam * x

    if x0 is None:
        x = torch.zeros_like(b)
    else:
        zero_rhs = (b == 0).all(dim=(-2, -1), keepdim=True)
        x = torch.where(zero_rhs, 0, x0)
    r = b.clone() if x0 is None else b - normal(x)
    p = r.clone()
    rr = _dot(r, r)
    tiny = torch.finfo(b.real.dtype).tiny
    norm = _dot(b, b).sqrt().clamp_min(tiny)
    active = rr.sqrt() / norm >= tol
    hit = torch.where(
        active.flatten(),
        torch.full((len(b),), iters, device=b.device, dtype=torch.int64),
        torch.zeros(len(b), device=b.device, dtype=torch.int64),
    )
    for step in range(iters):
        if not active.any():
            break
        normal_p = normal(p)
        denom = _dot(p, normal_p)
        if ((denom <= 0) & active).any() or not torch.isfinite(denom).all():
            raise RuntimeError("CG encountered a nonpositive or nonfinite curvature")
        alpha = torch.where(active, rr / denom.clamp_min(tiny), 0)
        x = x + alpha * p
        r = r - alpha * normal_p
        rr_new = _dot(r, r)
        relative = rr_new.sqrt() / norm
        converged = active & (relative < tol)
        hit = torch.where(converged.flatten(), step + 1, hit)
        next_active = active & ~converged
        beta = torch.where(next_active, rr_new / rr.clamp_min(tiny), 0)
        p = torch.where(next_active, r + beta * p, 0)
        rr, active = rr_new, next_active
    final_r = b - normal(x)
    residual = (_dot(final_r, final_r).sqrt() / norm).flatten()
    if not torch.isfinite(x).all() or not torch.isfinite(residual).all():
        raise RuntimeError("CG produced a nonfinite image or residual")
    return x, residual, hit
