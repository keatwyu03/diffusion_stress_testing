"""Dwivedi-Subba Rao portmanteau test for second-order stationarity.

Standalone and NOT wired into stationary_diagnosis.py or run_ticker. Pulled
out because the trispectrum correction needed real debugging (see history):
the raw scaling was off by T^2, the degenerate pairing terms have to be
zeroed BEFORE smoothing (not subtracted after), and a FIXED b4 does not
control the variance of kappa_r as T grows — b4 must scale with T (matching
b's T^(-1/3) rate) or the estimator produces negative correction denominators
at real sample sizes. That fix is verified (0/8 failures at T=6322 with
b4 = T^(-1/3), vs. repeated crashes with b4 fixed at 0.3) but not yet
integrated back into the main diagnostics.

Usage:
    from explore.test import dsr_portmanteau, holm, dsr_selftest
    dsr_selftest()  # chunked == unchunked, must pass before trusting output
    d = dsr_portmanteau(z, m_lags=4)   # b4 defaults to T^(-1/3)
"""

import numpy as np
import torch
from scipy.stats import chi2


def _bp_weights(T, bw):
    """Periodic Bartlett-Priestley weights over frequency-index offsets.

    The kernel argument is (w_k - w_j)/bw with w_k = 2 pi k / T, so support
    on [-1/2, 1/2] includes j when 2 pi |k-j| / T <= bw/2, i.e.
    |k-j| <= bw T / (4 pi). Positive, continuous, symmetric, normalised.
    """
    half = max(1, int(round(bw * T / (4.0 * np.pi))))
    u = np.arange(-half, half + 1) * (2.0 * np.pi / T) / bw
    w = np.maximum(1.5 * (1.0 - 4.0 * u ** 2), 0.0)
    return w / w.sum(), half


def _psmooth(v, w):
    """Circular convolution of v with symmetric odd-length weights w."""
    h = (len(w) - 1) // 2
    return np.convolve(np.concatenate([v[-h:], v, v[:h]]), w, mode="valid")


def dsr_portmanteau(z, m_lags=4, b=None, b4=None, ridge=0.0, row_chunk=512):
    """Dwivedi-Subba Rao portmanteau test for second-order stationarity.

    H0: {X_t} is second-order stationary. Under H0 the DFT ordinates at
    distinct canonical frequencies are asymptotically uncorrelated, so
    nonzero covariance between them is evidence against stationarity.

        J_T(w_k) = (2 pi T)^{-1/2} sum_t X_t exp(i t w_k),  w_k = 2 pi k / T

        c_T(r)   = (1/T) sum_k J(w_k) conj(J(w_{k+r}))
                          / sqrt( fhat(w_k) fhat(w_{k+r}) )

        T_m      = T sum_{r=1..m} |c_T(r)|^2 / (1 + kappa_r/2)  ~ chi2_{2m}

    Dividing by fhat standardises the ordinates for differing spectral power
    across frequencies; it does NOT substitute for Gaussianity. The per-lag
    corrections kappa_r do that, each estimated from the Brillinger-Rosenblatt
    trispectrum — four separate quantities, not one scalar excess-kurtosis
    proxy.

    SPECTRAL DENSITY (step 3)
        Bartlett-Priestley kernel, bandwidth b defaulting to T^{-1/3}. The
        paper prescribes neither; both are documented choices. `ridge` adds
        a small positive constant keeping fhat bounded away from zero.

    TRISPECTRUM (step 5)
        The correction needs f4 on the slice
            (nu1,nu2,nu3) = (lam_a, -lam_a - w_r, -lam_b),
        implied fourth frequency lam_b + w_r, so the raw fourth-order
        periodogram is

            I4_r(a,b) = J_a J_{-a-r} J_{-b} J_{b+r} * T / (2 pi).

        Each J already carries 1/sqrt(2 pi T), so the product of four carries
        1/(2 pi T)^2; multiplying by T/(2 pi) is what brings that down to the
        raw trispectrum's own 1/(2 pi T) scaling — dividing by (2 pi T) again
        would be off by a factor of T^2 and silently collapse every kappa_r
        toward zero. (This was a real bug in an earlier version.)

        f4 is a CUMULANT spectrum, so Gaussian pairing contributions must be
        removed from the raw fourth-moment product BEFORE smoothing — not
        after — so the exclusion is a hole the kernel then fills in from
        genuine neighbours, rather than a raw pairing value leaking into them.
        (Also a real bug in an earlier version: subtracting the pairing terms
        AFTER smoothing lets their raw values contaminate nearby frequencies.)
        On this slice:
          (1,2)(3,4): nu1+nu2 = -w_r != 0 for r >= 1 — NOT degenerate, kept;
          (1,3)(2,4): degenerate on a == b (mod T);
          (1,4)(2,3): degenerate on a + b + r == 0 (mod T).
        Each T x T slice is built, the two degenerate sets zeroed, then
        smoothed densely with a separable periodic 2-D kernel, divided by
        sqrt(fhat_a fhat_{a+r} fhat_b fhat_{b+r}), and summed.

    ROW CHUNKING
        A full slice is 640MB of complex128 at T=6322, and the raw copy, the
        smoothed copy and the denominator would be live at once. Rows are
        processed in blocks of `row_chunk`, each extended by half4 PERIODIC
        HALO ROWS above and below so the column-direction smoothing sees
        exactly the neighbours it would in the full array. Chunking changes
        peak memory only, never the estimate — checked by dsr_selftest().

    BANDWIDTH b4 — MUST SCALE WITH T
        `b4` is the trispectrum bandwidth and is REQUIRED (no silent default
        equal to b): the Brillinger-Rosenblatt estimator gives no automatic
        finite-sample rule. But a FIXED b4 does not control Var(kappa_r) as T
        grows — checked empirically: at b4=0.3 fixed, Gaussian white noise
        gives kappa_r std growing from ~1.4 (T=100) to ~2.8 (T=400), and at
        T=800-1600 individual draws produced 1 + kappa_r/2 <= 0, crashing the
        test. Scaling b4 = T^{-1/3} (matching b's rate) keeps the variance
        roughly flat across T and gave 0/8 failures at T=6322 in the same
        check. That default is used here for exactly that reason — pass a
        different b4 only after re-running the size check at your actual T.

    Returns dict with T, m_lags, b, b4, c (complex, per lag), kappa_r (real,
    per lag), correction_denominators, stat, df, p_value. Raises ValueError on
    a non-negligible imaginary part in kappa_r, a nonfinite kappa_r, or a
    nonpositive correction denominator — these indicate the estimator failed
    for this series/bandwidth rather than something to silently paper over.
    """
    x = np.asarray(z.detach().flatten().double().cpu().numpy(), dtype=np.float64)
    x = x - x.mean()                              # step 1: zero-mean
    T = len(x)

    if b4 is None:
        b4 = T ** (-1.0 / 3.0)

    # step 2. np.fft uses exp(-i t w); the paper's +i convention is its
    # conjugate. |c_T(r)| is unaffected, but the imaginary part then carries
    # the sign the write-up specifies.
    J = np.conj(np.fft.fft(x)) / np.sqrt(2.0 * np.pi * T)

    # step 3
    if b is None:
        b = T ** (-1.0 / 3.0)
    w2, half = _bp_weights(T, b)
    fhat = _psmooth(np.abs(J) ** 2, w2) + ridge
    fhat = np.maximum(fhat, np.finfo(np.float64).eps)

    w4, half4 = _bp_weights(T, b4)
    idx = np.arange(T)
    sqrt_f = np.sqrt(fhat)

    c = np.empty(m_lags, dtype=np.complex128)
    kappa = np.zeros(m_lags, dtype=np.complex128)

    for r in range(1, m_lags + 1):
        ar = (idx + r) % T

        # step 4
        c[r - 1] = np.mean(J * np.conj(J[ar]) / (sqrt_f * sqrt_f[ar]))

        # Column factor of the raw slice: g(b) = J_{-b} J_{b+r}, and the
        # column half of the denominator.
        col_raw = J[(-idx) % T] * J[(idx + r) % T]
        col_den = sqrt_f * sqrt_f[ar]

        acc = 0.0 + 0.0j
        for lo in range(0, T, row_chunk):
            hi = min(lo + row_chunk, T)
            rows = np.arange(lo - half4, hi + half4) % T      # periodic halo
            a = rows[:, None]
            bb = idx[None, :]

            # steps 5-6, densely on this row block
            raw = (J[a % T] * J[(-a - r) % T] *
                   col_raw[bb]) * (T / (2.0 * np.pi))

            # Zero the two degenerate pairing sets BEFORE smoothing, so the
            # kernel fills the hole from real neighbours rather than a raw
            # pairing value contaminating them.
            pairing_mask = (a == bb) | ((a + bb + r) % T == 0)
            raw[pairing_mask] = 0.0

            sm = np.empty_like(raw)
            for i in range(raw.shape[0]):
                sm[i, :] = _psmooth(raw[i, :], w4)
            sm = np.apply_along_axis(
                lambda col: np.convolve(col, w4, mode="valid"), 0, sm)

            arows = np.arange(lo, hi)
            den = (sqrt_f[arows][:, None] * sqrt_f[(arows + r) % T][:, None]
                   * col_den[None, :])
            block = sm / den

            acc += block.sum()
            del raw, sm, den, block

        kappa[r - 1] = (2.0 * np.pi / T ** 2) * acc

    # kappa_r is real in theory; a large imaginary part means the estimator
    # is misassembled, so check rather than silently discard it.
    kappa_r = np.real(kappa)
    kappa_im = np.imag(kappa)
    if np.any(np.abs(kappa_im) > 1e-6 * (1.0 + np.abs(kappa_r))):
        raise ValueError(
            f"trispectrum correction has a non-negligible imaginary part: "
            f"imag={kappa_im}, real={kappa_r}"
        )
    if not np.all(np.isfinite(kappa_r)):
        raise ValueError("Nonfinite fourth-order correction estimate.")

    denominator = 1.0 + 0.5 * kappa_r
    if np.any(denominator <= 0):
        raise ValueError(
            f"At least one fourth-order correction denominator is "
            f"nonpositive: {denominator}"
        )

    stat = float(T * np.sum(np.abs(c) ** 2 / denominator))
    df = 2 * m_lags
    return {
        "T": T,
        "m_lags": m_lags,
        "b": float(b),
        "b4": float(b4),
        "ridge": float(ridge),
        "kernel_half_width": half,
        "kernel_half_width_4": half4,
        "c": c,
        "kappa_r": kappa_r,
        "kappa_imag": kappa_im,
        "correction_denominators": denominator,
        "stat": stat,
        "df": df,
        "p_value": float(chi2.sf(stat, df)),
    }


def holm(p_values):
    """Holm step-down adjusted p-values, returned in the input order."""
    p = np.asarray(p_values, dtype=float)
    n = len(p)
    order = np.argsort(p)
    adj = np.empty(n)
    running = 0.0
    for i, k in enumerate(order):
        running = max(running, (n - i) * p[k])
        adj[k] = min(running, 1.0)
    return adj


def dsr_selftest(T=96, b4=0.3, seed=0, verbose=True):
    """Chunked dsr_portmanteau (row_chunk << T) must equal the unchunked
    version (row_chunk >= T, i.e. one pass with no splitting) at small T,
    to floating-point tolerance. Chunking is a memory optimisation only —
    this asserts it changes no number in the estimator.
    """
    rng = np.random.default_rng(seed)
    z = torch.tensor(rng.standard_normal(T), dtype=torch.float64)

    unchunked = dsr_portmanteau(z, b4=b4, row_chunk=T)
    for chunk in (7, 16, 33):
        chunked = dsr_portmanteau(z, b4=b4, row_chunk=chunk)
        for key in ("kappa_r", "c"):
            a, b_ = np.asarray(unchunked[key]), np.asarray(chunked[key])
            rel = np.max(np.abs(a - b_)) / max(np.max(np.abs(a)), 1e-300)
            if verbose:
                print(f"  row_chunk={chunk:<4} {key:<10} max rel diff = {rel:.3e}")
            assert rel < 1e-10, f"chunking changed {key}: rel diff {rel}"
        assert abs(unchunked["stat"] - chunked["stat"]) < 1e-8 * max(1.0, unchunked["stat"])
    if verbose:
        print(f"  dsr_selftest passed: chunked == unchunked at T={T}")
    return True


if __name__ == "__main__":
    dsr_selftest()
