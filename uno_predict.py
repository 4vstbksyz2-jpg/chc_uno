#!/usr/bin/env python
"""UNO for the Cahn-Hilliard-Cook equation -- prediction and diagnostics.

Loads the checkpoint written by uno_train.py, rolls it autoregressively on
unseen 128-box trajectories from t_start to t_end, and writes figures, a JSON
summary and the run log into Results/.

    python uno_predict.py
    python uno_predict.py --smoke

By default CONFIG["eval_offs"] evaluates every quench ch.py supplies
(-0.4, -0.2, 0.0, 0.2, 0.4) in one run: a full rollout, a figure pair and a
tuned stride schedule for each, then one summary table across all of them.
Narrow it to e.g. (0.0,) to evaluate the critical quench alone.

Ground truth is solved once and cached as .npz in Results/cache; later runs
reuse it.  All settings live in CONFIG below.
"""

from __future__ import annotations

import json
import math
import os
import sys
import time
from datetime import datetime
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from uno_train import (UNO, ground_truth, integrate,  # noqa: E402
                       make_schedule, set_precision, setup_output)

CONFIG: dict[str, Any] = dict(
    results_dir  = "Results",
    run_name     = "uno_ch",
    ckpt         = "",

    l            = 128,
    eval_offs    = (-0.4, -0.2, 0.0, 0.2, 0.4),
    test_seeds   = (99999, 54321, 77777, 31415, 27182,
                    61803, 41421, 57721, 66260, 86420),
    t_start      = 100.0,
    t_end        = 2000.0,
    threshold    = 0.80,
    snapshots    = (100.0, 500.0, 1000.0, 2000.0),
    solver_dt    = 0.01,
    save_every   = 4.0,

    alpha        = 0.20,
    dt_min       = 1.0,
    dt_max       = 40.0,

    do_tune      = True,
    tune_seeds   = (31337, 24601, 8675309, 20250, 42424,
                    90210, 23571, 55555, 48151, 62831),
    tune_report  = (500.0, 1000.0, 1500.0, 2000.0),
    tune_alphas  = (0.10, 0.20, 0.35, 0.50, 0.70),
    tune_dt_maxs = (30.0, 40.0, 45.0, 50.0),

    do_paper_figure   = True,
    paper_psi0        = (-0.4, -0.2, 0.0, 0.2, 0.4),
    paper_sizes       = (160, 192, 256),
    paper_size_off    = 0.0,
    paper_seeds       = (99999, 54321, 77777, 31415, 27182,
                         61803, 41421, 57721, 66260, 86420),
    paper_cells       = 4 * 128 * 128,
    paper_save_every  = 0.0,
    paper_markevery   = 2,

    do_transfer       = True,
    do_predictability = True,
    predictability_eps= (1e-4, 1e-3, 1e-2),
    do_drift          = True,
    do_speedup        = True,
    drift_dt          = 20.0,
    drift_times       = (100, 150, 200, 300, 400, 600, 800, 1100, 1400,
                         1700, 1900),
    L_train_max       = 18.8,

    device       = "cuda" if torch.cuda.is_available() else "cpu",
)

SMOKE: dict[str, Any] = dict(
    run_name="uno_ch_smoke", t_end=400.0,
    snapshots=(100.0, 200.0, 300.0, 400.0),
    test_seeds=(99999, 54321), do_tune=True, eval_offs=(0.0, -0.4),
    do_predictability=False, do_speedup=True, drift_times=(100, 200, 300),
    tune_seeds=(31337,), tune_report=(200.0, 400.0),
    tune_alphas=(0.2,), tune_dt_maxs=(40.0,),
    paper_psi0=(0.0,), paper_sizes=(160,), paper_seeds=(99999,))

TRUTH_C, PRED_C, MEAN_C = "#1b7f5f", "#7b3fa0", "#12457a"

PAPER_C = ("#1f2fd0", "#e11ec4", "#137a2e", "#c07000", "#00868b")
PAPER_M = ("o", "s", "^", "D", "v")
PAPER_RC: Any = {
    "font.family": "serif",
    "font.serif": ["cmr10", "DejaVu Serif"],
    "mathtext.fontset": "cm",
    "axes.formatter.use_mathtext": True,
    "axes.unicode_minus": False,
    "font.size": 10,
    "axes.labelsize": 12,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "axes.linewidth": 0.9,
    "lines.linewidth": 1.1,
    "lines.markersize": 4.0,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
    "xtick.minor.visible": True,
    "ytick.minor.visible": True,
    "xtick.major.size": 5.0,
    "ytick.major.size": 5.0,
    "xtick.minor.size": 2.8,
    "ytick.minor.size": 2.8,
    "xtick.major.width": 0.9,
    "ytick.major.width": 0.9,
    "legend.frameon": False,
    "legend.handlelength": 1.9,
    "legend.labelspacing": 0.35,
    "savefig.bbox": "tight",
}


def log(*a):
    print(*a, flush=True)


def free_energy_density(psi):
    gx = np.roll(psi, -1, -2) - psi
    gy = np.roll(psi, -1, -1) - psi
    return (-0.5 * psi ** 2 + 0.25 * psi ** 4
            + 0.5 * (gx ** 2 + gy ** 2)).mean(axis=(-2, -1))


def structure_factor(psi):
    """Angle-averaged S(k) of the fluctuation psi - <psi>."""
    psi = np.asarray(psi, dtype=np.float64)
    n = psi.shape[-1]
    fk = np.fft.fft2(psi - psi.mean(axis=(-2, -1), keepdims=True), norm="ortho")
    power = (fk.real ** 2 + fk.imag ** 2).mean(axis=tuple(range(psi.ndim - 2)))
    kx = 2.0 * np.pi * np.fft.fftfreq(n)
    kmag = np.sqrt(kx[:, None] ** 2 + kx[None, :] ** 2)
    dk, nb = 2.0 * np.pi / n, n // 2
    idx = np.clip((kmag / dk).astype(int), 0, nb - 1)
    cnt = np.bincount(idx.ravel(), minlength=nb)
    s = np.bincount(idx.ravel(), weights=power.ravel(), minlength=nb)
    ok = cnt > 0
    return ((np.arange(nb) + 0.5) * dk)[ok], (s[ok] / cnt[ok])


def domain_length(psi, method="zero"):
    """Domain scale L(t).  'zero' is quantised; use 'moment' for rates."""
    psi = np.asarray(psi, dtype=np.float64)
    if method == "zero":
        s = np.sign(psi - psi.mean(axis=(-2, -1), keepdims=True))
        cx = (s != np.roll(s, -1, -2)).mean(axis=(-2, -1))
        cy = (s != np.roll(s, -1, -1)).mean(axis=(-2, -1))
        return 1.0 / np.maximum(0.5 * (cx + cy), 1e-12)
    k, S = structure_factor(psi)
    if method == "moment":
        return 2.0 * np.pi * S.sum() / max((k * S).sum(), 1e-12)
    if method == "peak":
        return 2.0 * np.pi / k[int(np.argmax(S))]
    raise ValueError(method)


def pair_correlation(psi, rmax=None):
    psi = np.asarray(psi, dtype=np.float64)
    n = psi.shape[-1]
    fk = np.fft.fft2(psi - psi.mean(axis=(-2, -1), keepdims=True))
    corr = np.fft.ifft2(fk * np.conj(fk)).real / (n * n)
    corr = corr.mean(axis=tuple(range(psi.ndim - 2)))
    corr = corr / corr.flat[0]
    ax = np.minimum(np.arange(n), n - np.arange(n))
    r = np.sqrt(ax[:, None] ** 2 + ax[None, :] ** 2)
    rmax = rmax or n // 2
    idx = np.clip(np.rint(r).astype(int), 0, rmax)
    cnt = np.bincount(idx.ravel(), minlength=rmax + 1)
    s = np.bincount(idx.ravel(), weights=corr.ravel(), minlength=rmax + 1)
    ok = cnt > 0
    return np.arange(rmax + 1)[ok], s[ok] / cnt[ok]


def r2_score(pred, true):
    pred = np.asarray(pred, np.float64)
    true = np.asarray(true, np.float64)
    return float(1.0 - ((pred - true) ** 2).sum()
                 / max(((true - true.mean()) ** 2).sum(), 1e-30))


def first_below(times, r2, threshold):
    bad = np.nonzero(np.asarray(r2) < threshold)[0]
    return float(times[bad[0]]) if bad.size else None


@torch.no_grad()
def rollout(model, psi0, times, device, dtype, truth=None):
    """Autoregressive rollout over all seeds at once."""
    psi = torch.as_tensor(psi0, dtype=dtype, device=device)
    if psi.ndim == 3:
        psi = psi.unsqueeze(1)
    b = psi.shape[0]
    states = [psi[:, 0].float().cpu().numpy().copy()]
    r2 = [[1.0 if truth is None else r2_score(states[0][k], truth[0][k])
           for k in range(b)]]
    for i in range(1, len(times)):
        dt = torch.full((b,), float(times[i] - times[i - 1]), device=device,
                        dtype=dtype)
        psi = model(psi, dt)
        if not torch.isfinite(psi).all():
            log(f"  WARNING: non-finite state at t = {times[i]:g}; truncating")
            break
        states.append(psi[:, 0].float().cpu().numpy().copy())
        if truth is not None:
            r2.append([r2_score(states[-1][k], truth[i][k]) for k in range(b)])
    return np.array(states), np.array(r2)


def figure_rollout(res, snap_times, path, threshold, tag):
    seeds = list(res)
    n = len(snap_times)
    fig = plt.figure(figsize=(4.0 * n + 2.0, 14.5))
    gs = fig.add_gridspec(4, n + 1, height_ratios=[1.05, 1, 1, 1],
                          width_ratios=[1] * n + [0.06], hspace=0.42,
                          wspace=0.08)

    ax = fig.add_subplot(gs[0, :])
    t_common = np.asarray(res[seeds[0]]["times"])
    stack = np.stack([np.asarray(res[s]["r2"]) for s in seeds])
    t_common = t_common[:stack.shape[1]]
    mean_curve = stack.mean(axis=0)
    overall = float(stack.mean())

    ax.fill_between(t_common, stack.min(axis=0), stack.max(axis=0),
                    color=MEAN_C, alpha=0.15, lw=0,
                    label=f"min-max over {len(seeds)} seeds")
    ax.plot(t_common, mean_curve, color=MEAN_C, lw=2.8,
            label=f"mean $R^2$ over {len(seeds)} seeds")
    ax.axhline(overall, color=MEAN_C, ls=":", lw=1.8,
               label=f"mean of all $R^2$ values = {overall:.4f}")
    base = np.mean([np.asarray(res[s]["r2_persist"])[:len(t_common)]
                    for s in seeds], axis=0)
    ax.plot(t_common, base, color="0.55", lw=1.4, ls="-.",
            label="persistence baseline (mean)")
    ax.axhline(threshold, color="crimson", ls="--", lw=1.4,
               label=f"advisor threshold $R^2$ = {threshold:.2f}")

    below = np.nonzero(mean_curve < threshold)[0]
    t_cross_mean = float(t_common[below[0]]) if below.size else None
    if t_cross_mean is not None:
        ax.axvline(t_cross_mean, color="crimson", ls=":", lw=1.4)
        ax.annotate(f"mean crosses {threshold:.2f} at $t$ = {t_cross_mean:.0f}",
                    xy=(t_cross_mean, threshold), xytext=(10, -18),
                    textcoords="offset points", fontsize=9, color="crimson",
                    va="top")

    crossed = [res[s]["t_cross"] for s in seeds]
    never = [c for c in crossed if c is None]
    hit = sorted(c for c in crossed if c is not None)
    t_last = float(t_common[-1])
    if not hit:
        span = f"all {len(seeds)}/{len(seeds)} for the whole rollout"
    else:
        rng = (f"t = {hit[0]:.0f}-{hit[-1]:.0f}" if len(hit) > 1
               else f"t = {hit[0]:.0f}")
        span = (f"to {rng} for {len(hit)}/{len(seeds)}"
                + (f"; {len(never)}/{len(seeds)} never dropped below"
                   if never else ""))
    ax.set_xlabel("real time $t$")
    ax.set_ylabel("$R^2$ vs ground truth")
    ax.set_xlim(t_common[0], float(res[seeds[0]]["times"][-1]))
    ax.set_ylim(min(0.0, threshold - 0.1), 1.02)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower left", fontsize=9, ncol=2)
    ax.set_title(
        f"{tag} - full autoregressive rollout at l = {res[seeds[0]]['l']}, "
        f"t = {t_common[0]:.0f} → {t_last:.0f}   "
        f"(R² ≥ {threshold:.2f} held {span})", fontweight="bold")

    r = res[seeds[0]]
    rows = ("ground truth", f"{tag} prediction", "|error|")
    im_f = im_e = None
    for j, ts in enumerate(snap_times):
        k = int(np.argmin(np.abs(r["times"] - ts)))
        avail = k < len(r["states"])
        truth = r["truth"][k]
        for row in range(3):
            a = fig.add_subplot(gs[row + 1, j])
            a.set_xticks([]); a.set_yticks([]); a.set_box_aspect(1)
            if row == 0:
                im_f = a.imshow(truth, cmap="RdBu_r", vmin=-1, vmax=1)
            elif not avail:
                a.text(0.5, 0.5, "not reached" + os.linesep
                       + "(rollout truncated)",
                       ha="center", va="center", fontsize=11, color="crimson",
                       transform=a.transAxes)
                a.set_facecolor("0.95")
            elif row == 1:
                im_f = a.imshow(r["states"][k], cmap="RdBu_r", vmin=-1, vmax=1)
            else:
                im_e = a.imshow(np.abs(r["states"][k] - truth), cmap="magma",
                                vmin=0, vmax=1)
            if j == 0:
                a.set_ylabel(rows[row], fontsize=11)
            if row == 0:
                a.set_title(
                    f"$t={ts:g}$   (initial condition, given)" if k == 0 else
                    (f"$t={ts:g}$   ($R^2$ = {r['r2'][k]:.4f})" if avail
                     else f"$t={ts:g}$   (not reached)"), fontsize=11)
    if im_f is not None:
        fig.colorbar(im_f, cax=fig.add_subplot(gs[1:3, -1]))
    if im_e is not None:
        fig.colorbar(im_e, cax=fig.add_subplot(gs[3, -1]))
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def figure_physics(res, path, tag, snap_times):
    r = res[list(res)[0]]
    nv = len(r["states"])
    t = r["times"][:nv]
    fig, axes = plt.subplots(2, 3, figsize=(19, 11))
    fig.suptitle(f"{tag} — phase-ordering kinetics at $l={r['l']}$ "
                 f"(Puri, Kinetics of Phase Transitions, ch. 1)",
                 fontweight="bold", fontsize=15)

    ax = axes[0, 0]
    Lt = np.array([domain_length(x, "zero") for x in r["truth"][:nv]])
    Lp = np.array([domain_length(x, "zero") for x in r["states"]])
    ax.loglog(t, Lt, color=TRUTH_C, lw=2.2, label="ground truth")
    ax.loglog(t, Lp, color=PRED_C, lw=2.0, ls="--", label=tag)
    ax.loglog(t, Lt[0] * (t / t[0]) ** (1 / 3), "k:", lw=1.6,
              label="Lifshitz–Slyozov $t^{1/3}$")
    nt = np.polyfit(np.log(t), np.log(Lt), 1)[0]
    npd = np.polyfit(np.log(t), np.log(Lp), 1)[0]
    ax.set_title(f"Domain growth: truth $n$ = {nt:.3f}, {tag} $n$ = {npd:.3f}")
    ax.set_xlabel("$t$"); ax.set_ylabel("$L(t)$  [zero-crossing]")
    ax.legend(); ax.grid(alpha=0.3, which="both")

    show = [x for x in snap_times if x <= t[-1]]
    cols = plt.cm.viridis(np.linspace(0, 0.85, len(show)))

    ax = axes[0, 1]
    smax = 0.0
    for ts, c in zip(show, cols):
        k = int(np.argmin(np.abs(t - ts)))
        kk, S = structure_factor(r["truth"][k]); smax = max(smax, S.max())
        ax.loglog(kk, S, color=c, lw=2.0, label=f"$t={ts:g}$")
        kk, S = structure_factor(r["states"][k])
        ax.loglog(kk, S, color=c, lw=1.6, ls="--")
    kp = np.array([0.8, 2.5])
    ax.loglog(kp, 0.3 * smax * (kp / kp[0]) ** -3, "k:", lw=1.8,
              label="Porod $k^{-3}$")
    ax.set_ylim(smax * 1e-6, smax * 5)
    ax.set_title("Structure factor (solid: truth, dashed: prediction)")
    ax.set_xlabel("$k$"); ax.set_ylabel("$S(k,t)$")
    ax.legend(fontsize=9); ax.grid(alpha=0.3, which="both")

    ax = axes[0, 2]
    smax = 0.0
    for ts, c in zip(show, cols):
        k = int(np.argmin(np.abs(t - ts)))
        for arr, ls, L in ((r["truth"][k], "-", Lt[k]),
                           (r["states"][k], "--", Lp[k])):
            kk, S = structure_factor(arr)
            smax = max(smax, (S / L ** 2).max())
            ax.loglog(kk * L, S / L ** 2, color=c, lw=1.8, ls=ls,
                      label=f"$t={ts:g}$" if ls == "-" else None)
    ax.set_ylim(smax * 1e-6, smax * 5)
    ax.set_title("Dynamical scaling  $S = L^d f(kL)$  (Eq. 1.90)")
    ax.set_xlabel("$kL$"); ax.set_ylabel("$L^{-d}S(k,t)$")
    ax.legend(fontsize=9); ax.grid(alpha=0.3, which="both")

    ax = axes[1, 0]
    for ts, c in zip(show, cols):
        k = int(np.argmin(np.abs(t - ts)))
        for arr, ls, L in ((r["truth"][k], "-", Lt[k]),
                           (r["states"][k], "--", Lp[k])):
            rr, C = pair_correlation(arr)
            ax.plot(rr / L, C, color=c, lw=1.8, ls=ls,
                    label=f"$t={ts:g}$" if ls == "-" else None)
    ax.axhline(0, color="k", lw=0.8); ax.set_xlim(0, 3)
    ax.set_title("Scaling function  $C = g(r/L)$  (Eq. 1.88)")
    ax.set_xlabel("$r/L$"); ax.set_ylabel("$C(r,t)$")
    ax.legend(fontsize=9); ax.grid(alpha=0.3)

    ax = axes[1, 1]
    ft = np.array([free_energy_density(x) for x in r["truth"][:nv]])
    fp = np.array([free_energy_density(x) for x in r["states"]])
    ax.plot(t, ft, color=TRUTH_C, lw=2.2, label="ground truth")
    ax.plot(t, fp, color=PRED_C, lw=2.0, ls="--", label=tag)
    mono = bool(np.all(np.diff(fp) < 1e-12))
    ax.set_xscale("log")
    ax.set_title(f"Free-energy decay — prediction monotone: "
                 f"{'yes' if mono else 'no'}")
    ax.set_xlabel("$t$"); ax.set_ylabel(r"$\langle f\rangle$")
    ax.legend(); ax.grid(alpha=0.3)

    ax = axes[1, 2]
    for s in res:
        rr = res[s]
        nv2 = len(rr["states"])
        d = np.array([x.mean() for x in rr["states"]]) - rr["truth"][0].mean()
        ax.semilogy(rr["times"][:nv2], np.abs(d) + 1e-18, lw=1.6,
                    label=f"seed {s}")
    ax.set_title(r"Order-parameter conservation  "
                 r"$|\langle\psi\rangle(t)-\psi_0|$")
    ax.set_xlabel("$t$"); ax.set_ylabel("mass drift")
    ax.legend(fontsize=9); ax.grid(alpha=0.3, which="both")

    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return dict(n_truth=float(nt), n_pred=float(npd),
                free_energy_monotone=mono, L_truth_final=float(Lt[-1]),
                L_pred_final=float(Lp[-1]))


def evaluate(model, cfg, cache_dir, device, dtype, off, tag, out_dir,
             alpha, dt_max):
    """Full rollout to t_end for every test seed, then draw the figures."""
    seeds = list(cfg["test_seeds"])
    truth_list, times = [], None
    for seed in seeds:
        assert seed >= 20000, (f"seed {seed} overlaps ch.py's training "
                               f"[0,10000) / validation [10000,20000) ranges")
        gt = ground_truth(cache_dir, cfg["l"], off, seed, cfg["t_start"],
                          cfg["t_end"], cfg["save_every"], cfg["solver_dt"],
                          device)
        z = np.load(gt)
        gt_t, gt_f = z["times"], z["fields"]
        if times is None:
            times = make_schedule(cfg["t_start"], cfg["t_end"], alpha,
                                  cfg["dt_min"], dt_max,
                                  quantum=float(gt_t[1] - gt_t[0]),
                                  must_hit=cfg["snapshots"])
        idx = [int(np.argmin(np.abs(gt_t - x))) for x in times]
        truth_list.append(gt_f[idx].astype(np.float64))

    assert times is not None
    truth = np.stack(truth_list, axis=1)
    t0 = time.time()
    states, r2 = rollout(model, truth[0], times, device, dtype, truth=truth)
    wall = time.time() - t0
    n_valid = states.shape[0]

    res: dict[int, dict[str, Any]] = {}
    summary: dict[str, Any] = {"rollout_seconds": wall}
    for k, seed in enumerate(seeds):
        persist = np.array([r2_score(truth[0][k], truth[i][k])
                            for i in range(len(times))])
        r2_k = r2[:, k]
        t_cross = first_below(times[:n_valid], r2_k, cfg["threshold"])
        res[seed] = dict(times=times, truth=truth[:, k], states=states[:, k],
                         r2=r2_k, r2_persist=persist, l=cfg["l"],
                         t_cross=t_cross)
        summary[str(seed)] = dict(
            steps=len(times) - 1, reached=float(times[n_valid - 1]),
            r2_final=float(r2_k[-1]), r2_min=float(r2_k.min()),
            r2_mean=float(r2_k.mean()), t_cross_below_threshold=t_cross,
            r2_at={f"t={x:g}": float(r2_k[int(np.argmin(np.abs(times - x)))])
                   for x in cfg["snapshots"]})
        held = f"t={t_cross:g}" if t_cross else "the whole rollout"
        log(f"  seed {seed}: {len(times)-1} steps to t={times[n_valid-1]:g}, "
            f"final R2={r2_k[-1]:.4f}, mean R2={r2_k.mean():.4f}, "
            f"R2 >= {cfg['threshold']:.2f} until {held}")

    mean_curve = r2.mean(axis=1)
    t_cross_mean = first_below(times[:n_valid], mean_curve, cfg["threshold"])
    final_per_seed = r2[-1, :]
    mean_per_seed = r2.mean(axis=0)
    crossings = [summary[str(s)]["t_cross_below_threshold"] for s in seeds]
    reached = [c if c is not None else float(times[n_valid - 1])
               for c in crossings]
    summary["mean"] = dict(
        r2_final=float(mean_curve[-1]),
        r2_final_std=float(final_per_seed.std(ddof=1))
        if len(seeds) > 1 else 0.0,
        r2_mean=float(r2.mean()),
        r2_mean_std=float(mean_per_seed.std(ddof=1)) if len(seeds) > 1 else 0.0,
        t_cross_below_threshold=t_cross_mean,
        survives_mean=float(np.mean(reached)),
        survives_std=float(np.std(reached, ddof=1)) if len(seeds) > 1 else 0.0,
        n_never_below=int(sum(c is None for c in crossings)),
        n_seeds=len(seeds))
    log(f"  MEAN over {len(seeds)} seeds: final R2="
        f"{mean_curve[-1]:.4f} +/- {summary['mean']['r2_final_std']:.4f}, "
        f"mean of all R2={r2.mean():.4f} +/- "
        f"{summary['mean']['r2_mean_std']:.4f}")
    log(f"  survives to t = {summary['mean']['survives_mean']:.0f} +/- "
        f"{summary['mean']['survives_std']:.0f}; "
        f"{summary['mean']['n_never_below']}/{len(seeds)} never dropped below "
        f"{cfg['threshold']:.2f}; mean curve crosses at "
        f"{('t=%g' % t_cross_mean) if t_cross_mean else 'never'}")
    log(f"  rollout wall time {wall:.2f}s for {len(seeds)} seeds "
        f"({len(times)-1} steps each)")

    f1 = os.path.join(out_dir, f"{tag}_rollout_evaluation.png")
    f2 = os.path.join(out_dir, f"{tag}_phase_ordering_kinetics.png")
    figure_rollout(res, list(cfg["snapshots"]), f1, cfg["threshold"], tag)
    phys = figure_physics(res, f2, tag, list(cfg["snapshots"]))
    log(f"  physics: {phys}")
    log(f"  wrote {os.path.basename(f1)}, {os.path.basename(f2)}")
    return summary, phys


def tune(model, cfg, cache_dir, device, dtype, off):
    """Pick the stride schedule on the validation seeds, never a test seed."""
    seeds = list(cfg["tune_seeds"])
    report = [float(t) for t in cfg["tune_report"] if t <= cfg["t_end"] + 1e-9]
    if not report:
        report = [float(cfg["t_end"])]
    grid = ", ".join(f"{t:g}" for t in report)
    log(f"[tuning] stride schedule for psi0 = {off:+.1f} on {len(seeds)} "
        f"validation seed(s) {seeds} (held out of the reported test set)")
    log(f"[tuning] score = mean R2 at t = {grid}; a fixed grid, so schedules "
        f"of different length stay comparable")

    fields, gt_t = [], None
    for sd in seeds:
        z = np.load(ground_truth(cache_dir, cfg["l"], off, sd,
                                 cfg["t_start"], cfg["t_end"],
                                 cfg["save_every"], cfg["solver_dt"], device))
        gt_t = z["times"]
        fields.append(z["fields"])
    if gt_t is None:
        return cfg["alpha"], cfg["dt_max"]
    q = float(gt_t[1] - gt_t[0])

    head = "".join(f"{'R2(' + format(t, 'g') + ')':>10}" for t in report)
    log(f"  {'alpha':>7} {'dt_max':>7} {'steps':>6} {'score':>9}{head}")

    scores = []
    for a in cfg["tune_alphas"]:
        for dm in cfg["tune_dt_maxs"]:
            times = make_schedule(cfg["t_start"], cfg["t_end"], a,
                                  cfg["dt_min"], dm, quantum=q,
                                  must_hit=tuple(report))
            idx = [int(np.argmin(np.abs(gt_t - x))) for x in times]
            truth = np.stack([f[idx] for f in fields],
                             axis=1).astype(np.float64)
            _, r2b = rollout(model, truth[0], times, device, dtype,
                             truth=truth)
            r2 = r2b.mean(axis=1)
            tv = np.asarray(times, dtype=np.float64)
            at = [float(r2[min(int(np.argmin(np.abs(tv - tt))), len(r2) - 1)])
                  for tt in report]
            sc = float(np.mean(at))
            scores.append((sc, float(a), float(dm), len(times) - 1))
            log(f"  {a:7.3f} {dm:7.1f} {len(times) - 1:6d} {sc:9.4f}"
                + "".join(f"{v:10.4f}" for v in at))

    if not scores:
        return cfg["alpha"], cfg["dt_max"]
    best = max(scores, key=lambda x: x[0])
    log(f"  best: alpha={best[1]:g}, dt_max={best[2]:g} "
        f"({best[3]} steps) -> score {best[0]:.4f}")
    if best[2] >= max(cfg["tune_dt_maxs"]) - 1e-9:
        log("  [warn] dt_max sits at the top of the grid. Longer strides keep "
            "winning because they compound the per-step bias fewer times; do "
            "not widen the grid beyond the dt_max the model trained on or the "
            "step itself becomes an extrapolation.")
    if best[1] <= min(cfg["tune_alphas"]) + 1e-9:
        log("  [warn] alpha sits at the bottom of the grid")
    return best[1], best[2]


def sweep_curve(model, cfg, cache_dir, device, dtype, l, off, seeds,
                alpha, dt_max):
    """Mean R2(t) over seeds for one (box size, composition) combination."""
    fields, gt_t = [], None
    save_every = cfg["paper_save_every"] or cfg["save_every"]
    for seed in seeds:
        z = np.load(ground_truth(cache_dir, l, off, seed, cfg["t_start"],
                                 cfg["t_end"], save_every,
                                 cfg["solver_dt"], device, quiet=True))
        gt_t = z["times"]
        fields.append(z["fields"])
    if gt_t is None:
        raise ValueError("no seeds given")
    times = make_schedule(cfg["t_start"], cfg["t_end"], alpha, cfg["dt_min"],
                          dt_max, quantum=float(gt_t[1] - gt_t[0]))
    idx = [int(np.argmin(np.abs(gt_t - x))) for x in times]
    per = max(1, int(cfg["paper_cells"] // (l * l)))
    chunks = []
    for i in range(0, len(fields), per):
        truth = np.stack([f[idx] for f in fields[i:i + per]],
                         axis=1).astype(np.float64)
        _, r2b = rollout(model, truth[0], times, device, dtype, truth=truth)
        chunks.append(r2b)
    n = min(c.shape[0] for c in chunks)
    r2 = np.concatenate([c[:n] for c in chunks], axis=1)
    return np.asarray(times[:n], dtype=float), r2.mean(axis=1)


def _paper_panel(ax, curves, cfg, label):
    for i, (lab, t, r2) in enumerate(curves):
        ax.plot(t, r2, color=PAPER_C[i % len(PAPER_C)],
                marker=PAPER_M[i % len(PAPER_M)],
                markevery=cfg["paper_markevery"], label=lab)
    lo = min(float(r2.min()) for _, _, r2 in curves)
    lo = math.floor((lo - 0.008) / 0.05) * 0.05
    ax.set_ylim(lo, 1.008)
    ax.set_yticks(np.arange(lo, 1.0001, 0.05))
    ax.set_xlim(cfg["t_start"], cfg["t_end"])
    ax.set_ylabel("$R^2$")
    ax.text(0.955, 0.90, label, transform=ax.transAxes, ha="right",
            va="top")
    ax.legend(loc="lower left", borderaxespad=0.9,
              ncol=2 if len(curves) > 4 else 1,
              columnspacing=1.1)


def figure_paper(model, cfg, cache_dir, device, dtype, out_dir, alpha, dt_max):
    """FIG. 7 of the reference paper: R2(t) by composition and by box size."""
    seeds = list(cfg["paper_seeds"])
    log("")
    log(f"[figure] paper-style R2(t), mean over {len(seeds)} seeds {seeds}, "
        f"stride dt = clip({alpha:g} t, {cfg['dt_min']:g}, {dt_max:g})")

    top, bot, summary = [], [], {}
    for v in cfg["paper_psi0"]:
        t, r2 = sweep_curve(model, cfg, cache_dir, device, dtype, cfg["l"],
                            float(v), seeds, alpha, dt_max)
        pad = "" if v < 0 else r"\;"
        top.append((rf"$\psi_0 = {pad}{v:.1f}$", t, r2))
        summary[f"psi0={v:+.1f}"] = dict(l=cfg["l"], r2_final=float(r2[-1]),
                                         r2_mean=float(r2.mean()))
        log(f"  psi0 = {v:+.1f}, L = {cfg['l']:3d}: final R2 = {r2[-1]:.4f}, "
            f"mean R2 = {r2.mean():.4f}")
    for n in cfg["paper_sizes"]:
        t, r2 = sweep_curve(model, cfg, cache_dir, device, dtype, int(n),
                            float(cfg["paper_size_off"]), seeds, alpha, dt_max)
        bot.append((f"$L = {int(n)}$", t, r2))
        summary[f"L={int(n)}"] = dict(off=float(cfg["paper_size_off"]),
                                      r2_final=float(r2[-1]),
                                      r2_mean=float(r2.mean()))
        log(f"  psi0 = {cfg['paper_size_off']:+.1f}, L = {int(n):3d}: "
            f"final R2 = {r2[-1]:.4f}, mean R2 = {r2.mean():.4f}")

    with plt.rc_context(PAPER_RC):
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(3.6, 5.2), sharex=True)
        fig.subplots_adjust(hspace=0.09)
        _paper_panel(ax1, top, cfg, "(a)")
        _paper_panel(ax2, bot, cfg, "(b)")
        ax2.set_xlabel("$t$")
        ticks = [x for x in (500, 1000, 1500, 2000)
                 if cfg["t_start"] <= x <= cfg["t_end"]]
        if ticks:
            ax2.set_xticks(ticks)
        base = os.path.join(out_dir, f"{cfg['run_name']}_r2_sweep")
        fig.savefig(base + ".png", dpi=400)
        fig.savefig(base + ".pdf")
        plt.close(fig)
    log(f"  wrote {os.path.basename(base)}.png, "
        f"{os.path.basename(base)}.pdf")
    return summary


def diag_transfer(model, cfg, cache, device, dtype):
    log("")
    log("[diagnostic] 64 -> 128 TRANSFER")
    worst = 0.0
    for i in range(3):
        g = torch.Generator(device=device).manual_seed(i)
        psi = torch.randn(1, 1, 64, 64, generator=g, device=device,
                          dtype=dtype) * 0.6
        psi = psi - psi.mean()
        dt = torch.tensor([10.0], device=device, dtype=dtype)
        with torch.no_grad():
            a = model(psi, dt).repeat(1, 1, 2, 2)
            b = model(psi.repeat(1, 1, 2, 2), dt)
        worst = max(worst, float((a - b).abs().max()
                                 / (a - psi.repeat(1, 1, 2, 2)).abs().max()))
    log(f"  tile invariance |model(tile) - tile(model)| = {worst:.3e}   "
        f"{'PASS' if worst < 5e-3 else 'FAIL'}")

    z = np.load(cache)
    t, f = z["times"], z["fields"]
    log(f"  one-step R^2 at l={cfg['l']} (training only ever saw t <= 299)")
    log(f"    {'t0':>7} {'dt':>6} {'UNO R2':>10} {'persistence':>12} "
        f"{'gain':>9}")
    for t0 in (100.0, 1000.0, 1900.0):
        for dt in (5.0, 20.0, 40.0):
            i0 = int(np.argmin(np.abs(t - t0)))
            i1 = int(np.argmin(np.abs(t - (t0 + dt))))
            if i1 <= i0 or i1 >= len(t):
                continue
            x = torch.as_tensor(f[i0], dtype=dtype, device=device)[None, None]
            with torch.no_grad():
                p = model(x, torch.full((1,), dt, device=device, dtype=dtype))
            a = r2_score(p[0, 0].float().cpu().numpy(), f[i1])
            b = r2_score(f[i0], f[i1])
            flag = "" if t0 <= 299 else "   <- extrapolated in t"
            log(f"    {t0:7.0f} {dt:6.1f} {a:10.5f} {b:12.5f} "
                f"{a-b:+9.5f}{flag}")
    return dict(tile_invariance=worst)


def diag_predictability(cfg, cache, device):
    """How much rollout error is intrinsic rather than the model's."""
    log("")
    log("[diagnostic] INTRINSIC PREDICTABILITY (exact solver)")
    z = np.load(cache)
    t, f = z["times"], z["fields"]
    psi0 = f[0].astype(np.float64)
    marks = [x for x in (500.0, 1000.0, cfg["t_end"]) if x <= t[-1]]
    idx = [int(np.argmin(np.abs(t - x))) for x in marks]
    rng = np.random.default_rng(0)
    log("  " + f"{'rel. perturb':>13}  "
        + "  ".join(f"R2(t={x:g})".rjust(11) for x in marks))
    out = {}
    for eps in cfg["predictability_eps"]:
        n = rng.standard_normal(psi0.shape)
        n -= n.mean()
        n *= eps * psi0.std() / n.std()
        _, got = integrate(psi0 + n, float(t[-1] - t[0]), list(t - t[0]),
                           dt=cfg["solver_dt"], device=device)
        vals = [r2_score(got[i], f[i]) for i in idx]
        out[f"eps_{eps:g}"] = dict(zip([f"t{int(x)}" for x in marks], vals))
        log(f"  {eps:13.1e}  " + "  ".join(f"{v:11.5f}" for v in vals))
    return out


def diag_drift(model, cfg, cache, device, dtype):
    """Per-step coarsening-rate bias against domain size L."""
    log("")
    log("[diagnostic] PER-STEP COARSENING BIAS (continuous 'moment' estimator)")
    z = np.load(cache)
    t, f = z["times"], z["fields"]
    lmax = cfg["L_train_max"]
    log(f"  training data reaches only L ~ {lmax:.1f} (t = 299 on the 64-box)")
    log(f"  {'t':>7} {'L_truth':>9} {'dL_truth':>10} {'dL_model':>10} "
        f"{'bias %':>8}  in train range")
    rows = []
    for t0 in cfg["drift_times"]:
        i0 = int(np.argmin(np.abs(t - t0)))
        i1 = int(np.argmin(np.abs(t - (t0 + cfg["drift_dt"]))))
        if i1 <= i0 or i1 >= len(t):
            continue
        x0 = f[i0].astype(np.float64)
        with torch.no_grad():
            p = model(torch.as_tensor(x0, dtype=dtype,
                                      device=device)[None, None],
                      torch.full((1,), float(cfg["drift_dt"]), device=device,
                                 dtype=dtype))
        L0 = domain_length(x0, "moment")
        dt_ = domain_length(f[i1].astype(np.float64), "moment") - L0
        dm_ = domain_length(p[0, 0].float().cpu().numpy().astype(np.float64),
                            "moment") - L0
        bias = 100.0 * (dm_ - dt_) / abs(dt_) if abs(dt_) > 1e-9 else float("nan")
        rows.append((L0, bias))
        log(f"  {t0:7.0f} {L0:9.2f} {dt_:10.4f} {dm_:10.4f} {bias:8.1f}  "
            f"{'yes' if L0 <= lmax else 'NO -- extrapolated'}")
    ins = [b for L, b in rows if L <= lmax and np.isfinite(b)]
    out = [b for L, b in rows if L > lmax and np.isfinite(b)]
    res: dict[str, Any] = {}
    if ins:
        res["mean_bias_inside_pct"] = float(np.mean(ins))
        log(f"  mean bias inside  training L range ({len(ins)} pts): "
            f"{np.mean(ins):+.1f} %")
    if out:
        res["mean_bias_outside_pct"] = float(np.mean(out))
        log(f"  mean bias outside training L range ({len(out)} pts): "
            f"{np.mean(out):+.1f} %")
    return res


def diag_speedup(cfg, device, dtype, model, cache, rollout_seconds,
                 n_seeds, n_steps, times):
    """Surrogate wall time against the explicit solver it replaces."""
    log("")
    log("[diagnostic] SPEEDUP vs the explicit solver")
    span = cfg["t_end"] - cfg["t_start"]
    solver_steps = int(round(span / cfg["solver_dt"]))
    ic = np.random.default_rng(0).standard_normal((cfg["l"], cfg["l"])) * 0.05
    n_probe = 2000
    t0 = time.time()
    integrate(ic, n_probe * cfg["solver_dt"], [n_probe * cfg["solver_dt"]],
              dt=cfg["solver_dt"], device=device)
    probe = time.time() - t0
    solver_est = probe / n_probe * solver_steps

    psi0 = np.load(cache)["fields"][0][None].astype(np.float64)
    rollout(model, psi0, times[:3], device, dtype)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()
    rollout(model, psi0, times, device, dtype)
    if device.type == "cuda":
        torch.cuda.synchronize()
    single = time.time() - t0
    batched = rollout_seconds / max(1, n_seeds)

    log(f"  solver: {solver_steps:,} explicit steps of dt={cfg['solver_dt']} "
        f"-> {solver_est:.1f}s per trajectory (one at a time, "
        f"measured on {n_probe} steps)")
    lead = f"  UNO:    {n_steps} autoregressive steps -> "
    log(lead + f"{single:.3f}s per trajectory (one at a time)")
    log(" " * len(lead) + f"{batched:.3f}s per trajectory "
        f"(amortised over a batch of {n_seeds})")
    log(f"  speedup {solver_est / max(single, 1e-9):.0f}x like-for-like, "
        f"{solver_est / max(batched, 1e-9):.0f}x batched "
        f"({solver_steps / max(1, n_steps):.0f}x fewer steps)")
    log("  quote the like-for-like number: the solver probe is unbatched too. "
        "Both compare against the explicit dt=0.01 scheme, not a "
        "semi-implicit spectral solver, which would need far fewer steps.")
    return dict(solver_seconds_per_trajectory=float(solver_est),
                uno_seconds_per_trajectory=float(single),
                uno_seconds_per_trajectory_batched=float(batched),
                speedup=float(solver_est / max(single, 1e-9)),
                speedup_batched=float(solver_est / max(batched, 1e-9)),
                batch_size=int(n_seeds),
                solver_steps=solver_steps, uno_steps=n_steps)


def summary_table(rows, cfg):
    """One table across every quench evaluated in this run."""
    log("")
    log(f"[summary] all quenches, {len(cfg['test_seeds'])} test seeds, "
        f"l = {cfg['l']}, t = {cfg['t_start']:g} -> {cfg['t_end']:g}")
    log(f"  {'psi0':>6} {'alpha':>6} {'dt_max':>7} {'steps':>6} "
        f"{'mean R2':>18} {'final R2':>18} {'R2 >= thr to t':>18} "
        f"{'n never':>7} {'n_pred':>8}")
    for off, alpha, dt_max, steps, m, phys in rows:
        log(f"  {off:+6.1f} {alpha:6.2f} {dt_max:7.1f} {steps:6d} "
            f"{m['r2_mean']:9.4f} +-{m['r2_mean_std']:.4f} "
            f"{m['r2_final']:9.4f} +-{m['r2_final_std']:.4f} "
            f"{m['survives_mean']:11.0f} +-{m['survives_std']:<4.0f} "
            f"{m['n_never_below']:3d}/{m['n_seeds']:<3d} "
            f"{phys['n_pred']:8.4f}")
    best = max(rows, key=lambda r: r[4]["r2_mean"])
    worst = min(rows, key=lambda r: r[4]["r2_mean"])
    log(f"  best psi0 = {best[0]:+.1f} (mean R2 {best[4]['r2_mean']:.4f}), "
        f"worst psi0 = {worst[0]:+.1f} (mean R2 {worst[4]['r2_mean']:.4f})")


def main():
    cfg: dict[str, Any] = dict(CONFIG)
    smoke = "--smoke" in sys.argv
    if smoke:
        cfg.update(SMOKE)

    out_dir = setup_output(cfg, kind="predict")
    cache_dir = os.path.join(out_dir, "cache")
    device = torch.device(str(cfg["device"]))
    prec, dtype = set_precision()

    log("=" * 78)
    log(f" UNO / Cahn-Hilliard -- PREDICTION{'  [SMOKE]' if smoke else ''}   "
        f"{datetime.now():%Y-%m-%d %H:%M:%S}")
    log("=" * 78)
    if device.type == "cuda":
        p = torch.cuda.get_device_properties(device)
        log(f"[setup] {p.name}, {p.total_memory / 1e9:.1f} GB, "
            f"torch {torch.__version__}")
    log(f"[setup] device={device} precision={prec} dtype={dtype}")

    ckpt = cfg["ckpt"]
    if not ckpt:
        cands = [os.path.join(out_dir, f"{cfg['run_name']}_best.pt")]
        if smoke:
            cands.append(os.path.join(out_dir, f"{CONFIG['run_name']}_best.pt"))
        ckpt = next((c for c in cands if os.path.exists(c)), cands[0])
    if not os.path.exists(ckpt):
        raise SystemExit(f"checkpoint not found: {ckpt} -- run uno_train.py")
    ck = torch.load(ckpt, map_location=device, weights_only=False)
    model = UNO(**ck["arch"]).to(device, dtype=dtype).eval()
    model.load_state_dict(ck["model"])
    n_par = sum(p.numel() for p in model.parameters())
    log(f"[model] {os.path.basename(ckpt)}  {n_par:,} parameters  "
        f"arch={ck['arch']}")
    log(f"[model] training metrics: {ck.get('metrics')}")
    leak = sorted(set(cfg["tune_seeds"]) & set(cfg["test_seeds"]))
    assert not leak, (
        f"tune seeds {leak} are also test seeds; the stride schedule would "
        f"be selected on data it is then scored on")
    log(f"[eval] offs={list(cfg['eval_offs'])}  "
        f"{len(cfg['test_seeds'])} test seeds={list(cfg['test_seeds'])}")

    t0 = time.time()
    out: dict[str, Any] = dict(
        checkpoint=ckpt, arch=ck["arch"], precision=prec, l=cfg["l"],
        threshold=cfg["threshold"], schedule={})

    alpha, dt_max = cfg["alpha"], cfg["dt_max"]
    last_summary, n_steps, rows = None, 0, []
    for off in cfg["eval_offs"]:
        label = "critical" if abs(off) < 1e-9 else "off-critical"
        tag = (cfg["run_name"] if abs(off) < 1e-9
               else f"{cfg['run_name']}_off{off:+.1f}")
        log("")
        if cfg["do_tune"]:
            alpha, dt_max = tune(model, cfg, cache_dir, device, dtype, off)
            log("")
        out["schedule"][f"off{off:+.1f}"] = dict(
            alpha=alpha, dt_min=cfg["dt_min"], dt_max=dt_max)
        log(f"[rollout] {label} quench off={off:+.1f}, full horizon "
            f"t={cfg['t_start']:g} -> {cfg['t_end']:g}")
        s, p = evaluate(model, cfg, cache_dir, device, dtype, off, tag,
                        out_dir, alpha, dt_max)
        out[f"off{off:+.1f}"] = s
        out[f"off{off:+.1f}_physics"] = p
        last_summary = s
        n_steps = s[str(cfg["test_seeds"][0])]["steps"]
        rows.append((off, alpha, dt_max, n_steps, s["mean"], p))

    if len(rows) > 1:
        summary_table(rows, cfg)

    has_crit = any(abs(o) < 1e-9 for o in cfg["eval_offs"])
    diag_off = 0.0 if has_crit else cfg["eval_offs"][0]
    sel_gt = ground_truth(cache_dir, cfg["l"], diag_off,
                          cfg["tune_seeds"][0], cfg["t_start"], cfg["t_end"],
                          cfg["save_every"], cfg["solver_dt"], device)
    if cfg["do_paper_figure"]:
        out["r2_sweep"] = figure_paper(model, cfg, cache_dir, device, dtype,
                                       out_dir, alpha, dt_max)
    if cfg["do_transfer"]:
        out["transfer"] = diag_transfer(model, cfg, sel_gt, device, dtype)
    if cfg["do_predictability"]:
        out["predictability"] = diag_predictability(cfg, sel_gt, device)
    if cfg["do_drift"]:
        out["drift"] = diag_drift(model, cfg, sel_gt, device, dtype)
    if cfg["do_speedup"] and last_summary is not None:
        zq = np.load(sel_gt)["times"]
        sp_times = make_schedule(cfg["t_start"], cfg["t_end"], alpha,
                                 cfg["dt_min"], dt_max,
                                 quantum=float(zq[1] - zq[0]),
                                 must_hit=cfg["snapshots"])
        out["speedup"] = diag_speedup(
            cfg, device, dtype, model, sel_gt,
            last_summary["rollout_seconds"], len(cfg["test_seeds"]),
            len(sp_times) - 1, sp_times)

    with open(os.path.join(out_dir, f"{cfg['run_name']}_results.json"),
              "w") as f:
        json.dump(out, f, indent=2, default=float)
    log("")
    log(f"[done] total {time.time() - t0:.1f}s | {n_par:,} parameters | "
        f"results -> {out_dir}")


if __name__ == "__main__":
    main()
