#!/usr/bin/env python
"""UNO for the Cahn-Hilliard-Cook equation -- training.

Reads the dataset ch.py writes (test.npz) and writes everything into Results/.

    python uno_train.py
    python uno_train.py --smoke
    python uno_train.py --bench
    torchrun --nproc_per_node=4 uno_train.py

All settings live in CONFIG and PRECISION below.  See README.md for why the
architecture is built the way it is.
"""

from __future__ import annotations

import json
import math
import os
import sys
import time
import zipfile
from datetime import datetime
from typing import Any

if sys.platform != "win32":
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from numpy.lib import format as npformat


PRECISION = "fp32"

CONFIG: dict[str, Any] = dict(
    npz            = "test.npz",
    results_dir    = "Results",
    run_name       = "uno_ch",

    solver_dt      = 0.01,
    gap            = 10,
    off_values     = (-0.4, -0.2, 0.0, 0.2, 0.4),
    train_off      = None,

    t_min          = 35.0,
    dt_min         = 1.0,
    dt_max         = 50.0,
    in_ram         = True,
    data_on_gpu    = True,

    width          = 40,
    modes          = 20,
    levels         = 3,
    cond_dim       = 64,
    conserve_mass  = True,

    epochs         = 60,
    steps_per_epoch= 200,
    batch          = 128,
    lr             = 3.0e-3,
    lr_min         = 1.0e-5,
    unroll_stages  = ((0.00, 1), (0.08, 2), (0.19, 4)),
    stage_lr_mult  = 0.65,
    grad_steps     = 4,
    weight_decay   = 1.0e-4,
    grad_clip      = 1.0,
    ema_decay      = 0.999,
    noise          = 0.015,
    w_inc          = 0.25,
    w_grad         = 0.25,
    seed           = 0,

    val_seeds      = (31337, 24601, 8675309, 20250, 42424,
                      90210, 23571, 55555, 48151, 62831),
    val_l          = 128,
    val_t0         = 100.0,
    val_t1         = 2000.0,
    val_alpha      = 0.20,
    val_dt_max     = 40.0,
    val_save_every = 4.0,
    val_threshold  = 0.80,
    val_every      = 2,
    patience       = 0,
    min_delta      = 1.0e-4,

    resume         = True,
)

SMOKE: dict[str, Any] = dict(
    run_name="uno_ch_smoke", epochs=6, steps_per_epoch=25, val_every=3,
    patience=3, val_t1=400.0, val_save_every=10.0, val_seeds=(31337,),
    width=16, modes=8, batch=16, grad_steps=2, train_off=(0.0,))

_DTYPES = {"fp32": torch.float32, "tf32": torch.float32, "fp64": torch.float64}


def set_precision(mode=None):
    """Apply PRECISION globally.  Returns (mode, torch dtype)."""
    mode = (mode or PRECISION).lower()
    if mode not in _DTYPES:
        raise ValueError(
            f"PRECISION must be one of {sorted(_DTYPES)}, got {mode!r}")
    use_tf32 = mode == "tf32"
    torch.backends.cuda.matmul.allow_tf32 = use_tf32
    torch.backends.cudnn.allow_tf32 = use_tf32
    torch.set_float32_matmul_precision("high" if use_tf32 else "highest")
    return mode, _DTYPES[mode]


class Tee:
    """Duplicate a stream to a file so the run log lands in Results/."""

    def __init__(self, stream, fh):
        self.stream, self.fh = stream, fh

    def write(self, data):
        self.stream.write(data)
        self.fh.write(data)
        self.fh.flush()

    def flush(self):
        self.stream.flush()
        self.fh.flush()

    def isatty(self):
        return False


def init_distributed():
    """Returns (rank, world_size, local_rank, device); works with torchrun."""
    world = int(os.environ.get("WORLD_SIZE", 1))
    rank = int(os.environ.get("RANK", 0))
    local = int(os.environ.get("LOCAL_RANK", 0))
    if world > 1:
        dist.init_process_group(
            "nccl" if torch.cuda.is_available() else "gloo")
        if torch.cuda.is_available():
            torch.cuda.set_device(local)
    device = torch.device(f"cuda:{local}" if torch.cuda.is_available() else "cpu")
    return rank, world, local, device


def setup_output(cfg, rank=0, kind="train"):
    """Create Results/ (+cache) and tee stdout/stderr into the run log."""
    out = os.path.abspath(cfg["results_dir"])
    os.makedirs(out, exist_ok=True)
    os.makedirs(os.path.join(out, "cache"), exist_ok=True)
    stem = f"{cfg['run_name']}_{kind}"
    name = f"{stem}.log" if rank == 0 else f"{stem}.rank{rank}.log"
    fh = open(os.path.join(out, name), "a", buffering=1, encoding="utf-8")
    sys.stdout = Tee(sys.__stdout__, fh)
    sys.stderr = Tee(sys.__stderr__, fh)
    return out


def log(*a, rank=0):
    if rank == 0:
        print(*a, flush=True)


def make_ic(l, seed, off=0.0):
    """Reproduces PhaseOrdering.set_ic from ch.py."""
    np.random.seed(seed)
    ic = np.random.uniform(-0.1, 0.1, size=(l, l))
    return ic - ic.mean() + off


def _del_sq_t(psi):
    return (torch.roll(psi, 1, 0) + torch.roll(psi, -1, 0)
            + torch.roll(psi, 1, 1) + torch.roll(psi, -1, 1) - 4.0 * psi)


@torch.no_grad()
def integrate(ic, t_end, save_times, dt=0.01, device="cuda"):
    """Explicit Euler in float64, operation for operation as in ch.py."""
    psi = torch.as_tensor(np.ascontiguousarray(ic), dtype=torch.float64,
                          device=device)
    save = {int(round(t / dt)): i for i, t in enumerate(save_times)}
    out = np.empty((len(save_times),) + tuple(psi.shape), dtype=np.float32)
    if 0 in save:
        out[save[0]] = psi.cpu().numpy()
    for step in range(1, int(round(t_end / dt)) + 1):
        mu = psi**3 - psi - _del_sq_t(psi)
        psi = psi + _del_sq_t(mu) * dt
        j = save.get(step)
        if j is not None:
            out[j] = psi.cpu().numpy().astype(np.float32)
    return np.asarray(save_times, dtype=np.float64), out


def ground_truth(cache_dir, l, off, seed, t0, t1, save_every, dt, device,
                 rank=0, quiet=False):
    """Exact trajectory, cached to .npz and reused on later runs."""
    path = os.path.join(cache_dir, f"gt_l{l}_off{off:+.1f}_seed{seed}"
                                   f"_t{t0:g}-{t1:g}_ev{save_every:g}.npz")
    if os.path.exists(path):
        if not quiet:
            log(f"[data] ground truth cached: {os.path.basename(path)}",
                rank=rank)
        return path
    if rank == 0:
        log(f"[data] solving ground truth l={l} off={off:+.1f} seed={seed} "
            f"to t={t1:g} ...", rank=rank)
        t0_wall = time.time()
        times = np.arange(t0, t1 + 0.5 * save_every, save_every).round(6)
        t, f = integrate(make_ic(l, seed, off), t1, list(times), dt=dt,
                         device=device)
        np.savez(path, times=t, fields=f, seed=seed, off=off, l=l)
        log(f"[data]   -> {os.path.basename(path)} {f.shape} "
            f"({time.time() - t0_wall:.0f}s)", rank=rank)
    if dist.is_available() and dist.is_initialized():
        dist.barrier()
    return path


def frame_dt(cfg):
    return cfg["gap"] * cfg["solver_dt"]


def build_cache(cfg, out_dir, rank=0):
    """Stream test.npz into float32 memmaps; train_y is a shifted copy of X."""
    paths = {s: os.path.join(out_dir, "cache", f"{s}_X_f32.npy")
             for s in ("train", "val")}
    if all(os.path.exists(p) for p in paths.values()):
        return paths
    if rank == 0:
        src = cfg["npz"]
        if not os.path.exists(src):
            raise SystemExit(f"{src} not found -- run ch.py first")
        log(f"[data] reading {src} "
            f"({os.path.getsize(src) / 1e9:.2f} GB on disk)", rank=rank)
        with zipfile.ZipFile(src) as zf:
            for split, member in (("train", "train_X.npy"),
                                  ("val", "val_X.npy")):
                with zf.open(member) as f:
                    npformat.read_magic(f)
                    shape, fortran, dtype = npformat.read_array_header_1_0(f)
                    assert not fortran and dtype == np.float64
                    per = int(np.prod(shape[1:]))
                    arr = np.lib.format.open_memmap(
                        paths[split], mode="w+", dtype=np.float32, shape=shape)
                    for j in range(shape[0]):
                        buf = f.read(per * 8)
                        arr[j] = np.frombuffer(buf, np.float64).reshape(
                            shape[1:]).astype(np.float32)
                    arr.flush()
                    del arr
    if dist.is_available() and dist.is_initialized():
        dist.barrier()
    return paths


def off_of_trajectory(n_traj, off_values):
    """Per-trajectory mean order parameter, mirroring ch.py's array_split."""
    off = np.empty(n_traj, dtype=np.float64)
    for value, idx in zip(off_values,
                          np.array_split(np.arange(n_traj), len(off_values))):
        off[idx] = value
    return off


def select_trajectories(n_traj, cfg):
    """Indices whose quench is in cfg['train_off']; None selects all."""
    offs = off_of_trajectory(n_traj, cfg["off_values"])
    want = cfg["train_off"]
    if want is None:
        return np.arange(n_traj), offs, True
    want_arr = np.round(np.asarray(want, dtype=np.float64), 6)
    keep = np.nonzero(np.isin(np.round(offs, 6), want_arr))[0]
    if keep.size == 0:
        raise SystemExit(f"train_off={want} selects no trajectory; "
                         f"available off values are {sorted(set(offs))}")
    symmetric = set(np.round(-want_arr, 6)) == set(want_arr)
    return keep, offs[keep], symmetric


class PairSampler:
    """Draws (psi(t0), psi(t0+dt), ...) chains from the cached trajectories."""

    def __init__(self, path, cfg, device, dtype, seed):
        arr = np.load(path, mmap_mode="r")
        keep, self.offs, self.sign_ok = select_trajectories(arr.shape[0], cfg)
        data = np.ascontiguousarray(arr[keep]) if cfg["in_ram"] else arr[keep]
        self.kept = keep
        self.data = torch.from_numpy(np.ascontiguousarray(data))
        if cfg["data_on_gpu"] and device.type == "cuda":
            self.data = self.data.to(device)
        self.n_traj, self.n_frames = self.data.shape[:2]
        self.l = self.data.shape[-1]
        self.device, self.dtype = device, dtype
        self.rng = np.random.default_rng(seed)
        self.fdt = frame_dt(cfg)
        self.f_min = int(round((cfg["t_min"] - cfg["solver_dt"]) / self.fdt))
        self.df_min = max(1, int(round(cfg["dt_min"] / self.fdt)))
        self.df_max = int(round(cfg["dt_max"] / self.fdt))

    def max_dt(self, n_steps):
        """Largest dt a chain of n_steps fits into one training trajectory."""
        df_hi = max(self.df_min,
                    min(self.df_max,
                        (self.n_frames - 1 - self.f_min) // n_steps))
        return df_hi * self.fdt

    def sample(self, batch, n_steps=1, dt=None):
        df_hi = max(self.df_min,
                    min(self.df_max,
                        (self.n_frames - 1 - self.f_min) // n_steps))
        if dt is None:
            lo, hi = math.log(self.df_min), math.log(df_hi)
            df = np.clip(np.rint(np.exp(self.rng.uniform(lo, hi, batch))),
                         self.df_min, df_hi).astype(np.int64)
        else:
            df = np.minimum(np.full(batch, max(1, int(round(dt / self.fdt))),
                                    dtype=np.int64), df_hi)

        hi = self.n_frames - 1 - df * n_steps
        traj = self.rng.integers(0, self.n_traj, batch)
        f0 = self.rng.integers(self.f_min, np.maximum(hi, self.f_min + 1))

        src = self.data.device
        traj_i = torch.as_tensor(traj, device=src)
        chain = []
        for k in range(n_steps + 1):
            idx_i = torch.as_tensor(f0 + k * df, device=src)
            x = self.data[traj_i, idx_i]
            chain.append(x.to(self.device, dtype=self.dtype).unsqueeze(1))
        dt_t = torch.as_tensor(df * self.fdt, dtype=self.dtype,
                               device=self.device)
        return chain, dt_t

    def augment(self, chain, gen):
        """D4 lattice symmetry, plus psi -> -psi when the quench set allows."""
        k = int(torch.randint(0, 4, (1,), generator=gen, device=self.device))
        if k:
            chain = [torch.rot90(c, k, dims=(-2, -1)) for c in chain]
        if torch.rand((), generator=gen, device=self.device) < 0.5:
            chain = [torch.flip(c, dims=(-1,)) for c in chain]
        if self.sign_ok and torch.rand((), generator=gen,
                                       device=self.device) < 0.5:
            chain = [-c for c in chain]
        return chain


class SpectralConv2d(nn.Module):
    """Fourier layer whose kernel is defined on physical wavenumbers."""

    def __init__(self, in_ch, out_ch, modes1, modes2, n_ref):
        super().__init__()
        self.in_ch, self.out_ch = in_ch, out_ch
        self.modes1, self.modes2, self.n_ref = modes1, modes2, n_ref
        scale = 1.0 / (in_ch * out_ch) ** 0.5
        self.weight = nn.Parameter(
            scale * torch.randn(in_ch, out_ch, 2 * modes1 + 1, modes2, 2))
        self._cache = None

    def _runtime_weight(self, s, n, device, dtype):
        key = (s, n, device, dtype)
        if self._cache is not None and self._cache[0] == key \
                and not torch.is_grad_enabled():
            return self._cache[1]

        f1 = int(round(self.modes1 * s))
        f2 = int(round((self.modes2 - 1) * s))
        assert 2 * f1 < n and f2 < n // 2, (
            f"mode budget {self.modes1}/{self.modes2} too large for grid {n} "
            f"(s={s})")

        real_dtype = (torch.float64 if dtype == torch.complex128
                      else torch.float32)
        w = self.weight.permute(4, 0, 1, 2, 3).reshape(
            2 * self.in_ch * self.out_ch, 1,
            2 * self.modes1 + 1, self.modes2).to(real_dtype)
        w = F.interpolate(w, size=(2 * f1 + 1, f2 + 1), mode="bilinear",
                          align_corners=True)
        w = w.reshape(2, self.in_ch, self.out_ch, 2 * f1 + 1, f2 + 1)
        w = torch.complex(w[0], w[1]).to(device=device, dtype=dtype)
        w = torch.cat([w[:, :, f1:], w[:, :, :f1]], dim=2)
        if not torch.is_grad_enabled():
            self._cache = (key, w)
        return w

    def forward(self, x, s):
        b, c, n, m = x.shape
        x_ft = torch.fft.rfft2(x, norm="forward")
        w = self._runtime_weight(s, n, x_ft.device, x_ft.dtype)
        cols, f1 = w.shape[3], (w.shape[2] - 1) // 2

        blk = torch.cat([x_ft[:, :, :f1 + 1, :cols],
                         x_ft[:, :, n - f1:, :cols]], dim=2)
        out = torch.einsum("bixy,ioxy->boxy", blk, w)

        if out.shape[2] > 1:
            rows = out.shape[2]
            perm = torch.arange(rows, device=out.device)
            perm = torch.where(perm == 0, perm, rows - perm)
            c0 = out[:, :, :, 0]
            c0 = 0.5 * (c0 + c0[:, :, perm].conj())
            out = torch.cat([c0.unsqueeze(-1), out[:, :, :, 1:]], dim=-1)

        full = torch.zeros(b, self.out_ch, n, n // 2 + 1, dtype=x_ft.dtype,
                           device=x_ft.device)
        full[:, :, :f1 + 1, :cols] = out[:, :, :f1 + 1]
        full[:, :, n - f1:, :cols] = out[:, :, f1 + 1:]
        return torch.fft.irfft2(full, s=(n, n), norm="forward")


class FiLM(nn.Module):
    def __init__(self, cond_dim, ch):
        super().__init__()
        self.fc = nn.Linear(cond_dim, 2 * ch)
        nn.init.zeros_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

    def forward(self, x, cond):
        g, b = self.fc(cond).chunk(2, dim=-1)
        return x * (1.0 + g[:, :, None, None]) + b[:, :, None, None]


class UNOBlock(nn.Module):
    """Fourier path plus local 3x3 path plus residual, modulated by dt."""

    def __init__(self, in_ch, out_ch, m1, m2, n_ref, cond_dim):
        super().__init__()
        self.spec = SpectralConv2d(in_ch, out_ch, m1, m2, n_ref)
        self.local = nn.Conv2d(in_ch, out_ch, 3, padding=1,
                               padding_mode="circular")
        self.point = nn.Conv2d(in_ch, out_ch, 1)
        self.film = FiLM(cond_dim, out_ch)
        self.mix = nn.Conv2d(out_ch, out_ch, 1)
        self.skip = nn.Identity() if in_ch == out_ch else \
            nn.Conv2d(in_ch, out_ch, 1, bias=False)

    def forward(self, x, cond, s):
        h = self.spec(x, s) + self.local(x) + self.point(x)
        h = self.mix(F.gelu(self.film(h, cond)))
        return F.gelu(self.skip(x) + h)


def _dt_embedding(dt, dim):
    half = dim // 2
    logdt = torch.log(dt.clamp_min(1e-6))
    freqs = torch.exp(torch.linspace(0.0, math.log(64.0), half,
                                     device=dt.device, dtype=dt.dtype))
    ang = logdt[:, None] * freqs[None, :]
    return torch.cat([torch.sin(ang), torch.cos(ang), logdt[:, None]], dim=-1)


class UNO(nn.Module):
    """psi(t) -> psi(t + dt); box-size agnostic and mass conserving."""

    def __init__(self, width=32, modes=16, n_ref=64, levels=3, cond_dim=64,
                 conserve_mass=True):
        super().__init__()
        self.n_ref, self.levels = n_ref, levels
        self.conserve_mass = conserve_mass
        emb = 2 * (cond_dim // 2) + 1
        self.cond_in = emb
        self.cond = nn.Sequential(nn.Linear(emb, cond_dim), nn.GELU(),
                                  nn.Linear(cond_dim, cond_dim), nn.GELU())

        widths = [int(width * 1.5 ** i) for i in range(levels + 1)]
        self.mode_list = [max(2, min(modes // 2 ** i,
                                     (n_ref // 2 ** i - 1) // 2))
                          for i in range(levels + 1)]

        self.lift = nn.Conv2d(4, widths[0], 1)
        self.down = nn.ModuleList([
            UNOBlock(widths[i], widths[i], self.mode_list[i],
                     self.mode_list[i] + 1, n_ref, cond_dim)
            for i in range(levels)])
        self.reduce = nn.ModuleList([nn.Conv2d(widths[i], widths[i + 1], 1)
                                     for i in range(levels)])
        self.mid = UNOBlock(widths[levels], widths[levels],
                            self.mode_list[levels], self.mode_list[levels] + 1,
                            n_ref, cond_dim)
        self.expand = nn.ModuleList([nn.Conv2d(widths[i + 1], widths[i], 1)
                                     for i in range(levels)])
        self.smooth = nn.ModuleList([
            nn.Conv2d(widths[i], widths[i], 3, padding=1,
                      padding_mode="circular") for i in range(levels)])
        self.up = nn.ModuleList([
            UNOBlock(2 * widths[i], widths[i], self.mode_list[i],
                     self.mode_list[i] + 1, n_ref, cond_dim)
            for i in range(levels)])
        head_out = nn.Conv2d(widths[0], 1, 1)
        nn.init.zeros_(head_out.weight)
        assert head_out.bias is not None
        nn.init.zeros_(head_out.bias)
        self.head = nn.Sequential(nn.Conv2d(widths[0], widths[0], 1), nn.GELU(),
                                  head_out)

        self.log_gain = nn.Parameter(torch.tensor(math.log(0.012)))
        self.gain_exp = nn.Parameter(torch.tensor(1.0))

    @staticmethod
    def _features(psi):
        lap = (torch.roll(psi, 1, -2) + torch.roll(psi, -1, -2)
               + torch.roll(psi, 1, -1) + torch.roll(psi, -1, -1) - 4.0 * psi)
        mean = psi.mean(dim=(-2, -1), keepdim=True).expand_as(psi)
        return torch.cat([psi, psi ** 3, lap, mean], dim=1)

    def forward(self, psi, dt):
        if dt.ndim == 0:
            dt = dt.expand(psi.shape[0])
        cond = self.cond(_dt_embedding(dt.to(psi.dtype), self.cond_in - 1))
        s = psi.shape[-1] / self.n_ref

        x = self.lift(self._features(psi))
        skips = []
        for i in range(self.levels):
            x = self.down[i](x, cond, s)
            skips.append(x)
            x = F.avg_pool2d(self.reduce[i](x), 2)
        x = self.mid(x, cond, s)
        for i in reversed(range(self.levels)):
            x = F.interpolate(self.expand[i](x), scale_factor=2, mode="nearest")
            x = self.smooth[i](x)
            x = self.up[i](torch.cat([x, skips[i]], dim=1), cond, s)

        gain = torch.exp(self.log_gain
                         + self.gain_exp * torch.log(dt.clamp_min(1e-6)))
        inc = self.head(x) * gain[:, None, None, None].to(x.dtype)
        if self.conserve_mass:
            inc = inc - inc.mean(dim=(-2, -1), keepdim=True)
        return psi + inc


def _sq(x):
    return (x ** 2).sum(dim=(1, 2, 3))


def r2_loss(pred, true):
    den = _sq(true - true.mean(dim=(1, 2, 3), keepdim=True))
    return (_sq(pred - true) / den.clamp_min(1e-12)).mean()


def increment_loss(pred, true, inp):
    return (_sq(pred - true) / _sq(true - inp).clamp_min(1e-12)).mean()


def _fwd_diff(z):
    return torch.roll(z, -1, -2) - z, torch.roll(z, -1, -1) - z


def gradient_loss(pred, true):
    px, py = _fwd_diff(pred)
    tx, ty = _fwd_diff(true)
    return ((_sq(px - tx) + _sq(py - ty))
            / (_sq(tx) + _sq(ty)).clamp_min(1e-12)).mean()


def add_noise(psi, sigma):
    if sigma <= 0:
        return psi
    n = torch.randn_like(psi) * sigma
    return psi + (n - n.mean(dim=(-2, -1), keepdim=True))


class EMA:
    def __init__(self, model, decay):
        self.decay = decay
        self.shadow = {k: v.detach().clone()
                       for k, v in model.state_dict().items()
                       if v.dtype.is_floating_point}

    @torch.no_grad()
    def update(self, model):
        for k, v in model.state_dict().items():
            if k in self.shadow:
                self.shadow[k].mul_(self.decay).add_(v.detach(),
                                                     alpha=1 - self.decay)

    def copy_to(self, model):
        sd = model.state_dict()
        for k, v in self.shadow.items():
            sd[k].copy_(v)


def make_schedule(t0, t1, alpha, dt_min, dt_max, quantum=1.0, must_hit=()):
    """Times visited by an autoregressive rollout, stride ~ alpha * t."""
    ts, t = [float(t0)], float(t0)
    while t < t1 - 1e-9:
        dt = min(max(alpha * t, dt_min), dt_max)
        dt = max(quantum, round(dt / quantum) * quantum)
        t = min(t1, t + dt)
        ts.append(round(t, 6))
    ts += [float(m) for m in must_hit if t0 < m < t1]
    return np.array(sorted(set(ts)))


@torch.no_grad()
def eval_one_step(model, sampler, dts=(1.0, 10.0, 40.0), batch=8, reps=3):
    model.eval()
    acc = []
    for dt in dts:
        for _ in range(reps):
            chain, dt_t = sampler.sample(batch, 1, dt=dt)
            pred, true = model(chain[0], dt_t), chain[1]
            acc.append(float(r2_loss(pred, true)))
    model.train()
    return float(np.mean(acc))


@torch.no_grad()
def eval_rollout(model, gt_paths, cfg, device, dtype):
    """Full 128-box rollout on the held-out seeds.

    Returns the mean R^2 over every seed and time (the selection score), the
    mean survival time, and the per-seed survival times.
    """
    model.eval()
    truth_list, times = [], None
    for path in gt_paths:
        z = np.load(path)
        t, f = z["times"], z["fields"]
        if times is None:
            times = make_schedule(cfg["val_t0"], cfg["val_t1"],
                                  cfg["val_alpha"], cfg["dt_min"],
                                  cfg["val_dt_max"],
                                  quantum=float(t[1] - t[0]))
        idx = [int(np.argmin(np.abs(t - x))) for x in times]
        truth_list.append(f[idx].astype(np.float64))

    assert times is not None
    truth = np.stack(truth_list, axis=1)
    psi = torch.as_tensor(truth[0], dtype=dtype, device=device).unsqueeze(1)
    b = psi.shape[0]
    r2 = np.ones((len(times), b))
    for i in range(1, len(times)):
        dt = torch.full((b,), float(times[i] - times[i - 1]), device=device,
                        dtype=dtype)
        psi = model(psi, dt)
        if not torch.isfinite(psi).all():
            r2[i:] = -1.0
            break
        tr = torch.as_tensor(truth[i], dtype=dtype, device=device).unsqueeze(1)
        num = _sq(psi - tr)
        den = _sq(tr - tr.mean(dim=(1, 2, 3), keepdim=True))
        r2[i] = (1 - num / den).double().cpu().numpy()

    survive = []
    for k in range(b):
        bad = np.nonzero(r2[:, k] < cfg["val_threshold"])[0]
        survive.append(float(times[bad[0]]) if bad.size else float(times[-1]))
    model.train()
    return float(r2.mean()), float(np.mean(survive)), survive


def tuple_of(x):
    if isinstance(x, (list, tuple)):
        return tuple(tuple_of(v) for v in x)
    return x


def unroll_for(epoch, cfg):
    """Unroll length and stage index for this epoch."""
    total = cfg["epochs"]
    stage, unroll = 0, 1
    for i, (frac, n) in enumerate(cfg["unroll_stages"]):
        if epoch >= 1 + int(frac * total):
            stage, unroll = i, n
    return unroll, stage


def benchmark(cfg, device, dtype):
    """Time this GPU across the precision modes and check tile invariance."""
    log("=" * 78)
    log(" BENCHMARK")
    if device.type == "cuda":
        p = torch.cuda.get_device_properties(device)
        log(f" {p.name}  {p.total_memory / 2**30:.0f} GiB  "
            f"{p.multi_processor_count} SMs")
    log("=" * 78)
    bs = int(cfg["batch"])
    log(f" width={cfg['width']} modes={cfg['modes']} levels={cfg['levels']} "
        f"batch={bs}")
    log(f" {'mode':>6} {'fwd+bwd':>12} {'samples/s':>11} {'peak mem':>10} "
        f"{'tile-inv':>11}")
    for mode in ("fp32", "tf32", "fp64"):
        if device.type != "cuda" and mode == "tf32":
            continue
        _, dt_ = set_precision(mode)
        torch.manual_seed(0)
        net = UNO(width=cfg["width"], modes=cfg["modes"], n_ref=64,
                  levels=cfg["levels"], cond_dim=cfg["cond_dim"],
                  conserve_mass=cfg["conserve_mass"]).to(device, dtype=dt_)
        x = (torch.randn(bs, 1, 64, 64, device=device, dtype=dt_) * 0.5)
        x = x - x.mean(dim=(-2, -1), keepdim=True)
        d = torch.full((bs,), 10.0, device=device, dtype=dt_)
        for _ in range(5):
            net.zero_grad(set_to_none=True)
            net(x, d).square().mean().backward()
        if device.type == "cuda":
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        for _ in range(20):
            net.zero_grad(set_to_none=True)
            net(x, d).square().mean().backward()
        if device.type == "cuda":
            torch.cuda.synchronize()
        per = (time.time() - t0) / 20
        peak = (torch.cuda.max_memory_allocated() / 2**30
                if device.type == "cuda" else 0.0)
        net.eval()
        with torch.no_grad():
            g = torch.Generator(device=device).manual_seed(0)
            q = torch.randn(1, 1, 64, 64, generator=g, device=device,
                            dtype=dt_) * 0.6
            q = q - q.mean()
            d1 = torch.tensor([10.0], device=device, dtype=dt_)
            aa = net(q, d1).repeat(1, 1, 2, 2)
            bb = net(q.repeat(1, 1, 2, 2), d1)
            tile = float((aa - bb).abs().max()
                         / (aa - q.repeat(1, 1, 2, 2)).abs().max())
        log(f" {mode:>6} {per*1e3:10.1f} ms {bs/per:11.0f} {peak:9.2f} GiB "
            f"{tile:11.3e}")
        del net
        if device.type == "cuda":
            torch.cuda.empty_cache()
    log("")
    log(" If fwd+bwd is well under ~20 ms the run is launch-bound: raise batch")
    log(" or widen the model before changing anything else.")


def train(cfg, device, dtype, prec, rank, world, local, out_dir, smoke):
    torch.manual_seed(cfg["seed"] + rank)
    np.random.seed(cfg["seed"] + rank)
    torch.backends.cudnn.benchmark = True
    if "SLURM_CPUS_PER_TASK" in os.environ:
        torch.set_num_threads(max(1, int(os.environ["SLURM_CPUS_PER_TASK"])))

    gpu_gb = 0.0
    if device.type == "cuda":
        p = torch.cuda.get_device_properties(device)
        gpu_gb = p.total_memory / 2**30
        log(f"[setup] {p.name}, {p.total_memory / 1e9:.1f} GB, "
            f"torch {torch.__version__}", rank=rank)
    if prec == "fp64":
        log("[setup] note: test.npz is float32, so fp64 raises cost "
            "(~5x fewer FLOP/s on an A100, 2x memory) without adding "
            "target accuracy; tf32 or fp32 train the same model faster",
            rank=rank)
        if 0.0 < gpu_gb < 12.0:
            log(f"[setup] WARNING: fp64 on a {gpu_gb:.1f} GiB device is "
                f"likely to run out of memory (fp64 peaked at 7.6 GB on this "
                f"config). Set PRECISION='fp32' or lower batch/width, or "
                f"request a full A100.", rank=rank)
    log(f"[setup] device={device} precision={prec} dtype={dtype} "
        f"world_size={world}", rank=rank)
    log(f"[setup] results -> {out_dir}", rank=rank)

    paths = build_cache(cfg, out_dir, rank)
    train_s = PairSampler(paths["train"], cfg, device, dtype,
                          cfg["seed"] + 100 + rank)
    val_s = PairSampler(paths["val"], cfg, device, dtype,
                        cfg["seed"] + 200 + rank)
    gb = train_s.data.numel() * 4 / 1e9
    log(f"[data] train {tuple(train_s.data.shape)}  "
        f"val {tuple(val_s.data.shape)}  float32, {gb:.2f} GB", rank=rank)
    sel = cfg["train_off"]
    log(f"[data] train_off={sel if sel is not None else 'all'} -> "
        f"trajectories {[int(i) for i in train_s.kept]} "
        f"(off={sorted(set(train_s.offs.tolist()))})", rank=rank)
    log(f"[data] sign-flip augmentation "
        f"{'on' if train_s.sign_ok else 'off'}, sampling from "
        f"{train_s.data.device}", rank=rank)

    cache_dir = os.path.join(out_dir, "cache")
    gt_paths = [ground_truth(cache_dir, cfg["val_l"], 0.0, s, cfg["val_t0"],
                             cfg["val_t1"], cfg["val_save_every"],
                             cfg["solver_dt"], device, rank, quiet=True)
                for s in cfg["val_seeds"]]
    log(f"[val] l={cfg['val_l']} ground truth for seeds "
        f"{list(cfg['val_seeds'])}", rank=rank)

    raw = UNO(width=cfg["width"], modes=cfg["modes"], n_ref=train_s.l,
              levels=cfg["levels"], cond_dim=cfg["cond_dim"],
              conserve_mass=cfg["conserve_mass"]).to(device, dtype=dtype)
    n_par = sum(p.numel() for p in raw.parameters() if p.requires_grad)
    log(f"[model] {n_par:,} parameters, modes/level {raw.mode_list}, "
        f"dtype={dtype}", rank=rank)
    log(f"[model] conserve_mass={cfg['conserve_mass']} dt_conditioned=True "
        f"residual_prediction=True", rank=rank)

    ceil = [(n, train_s.max_dt(n)) for _, n in cfg["unroll_stages"]]
    log("[data] max trainable dt by unroll: "
        + ", ".join(f"{n} -> {d:.1f}" for n, d in ceil), rank=rank)
    short = [(n, d) for n, d in ceil if d < cfg["dt_max"] - 1e-9]
    if short:
        log(f"[warn] chain length caps dt below cfg dt_max={cfg['dt_max']:g} "
            f"for unroll {[n for n, _ in short]}: a t <= "
            f"{(train_s.n_frames - 1) * train_s.fdt:.0f} trajectory has no "
            f"room for longer chains at full stride. Those stages train a "
            f"shorter-dt regime than the rollout uses.", rank=rank)

    arch: dict[str, Any] = dict(
        width=cfg["width"], modes=cfg["modes"], n_ref=train_s.l,
        levels=cfg["levels"], cond_dim=cfg["cond_dim"],
        conserve_mass=cfg["conserve_mass"])
    sched: dict[str, Any] = dict(
        epochs=cfg["epochs"], steps_per_epoch=cfg["steps_per_epoch"],
        lr=cfg["lr"], unroll_stages=tuple_of(cfg["unroll_stages"]))
    state: dict[str, Any] = {}
    state_path = os.path.join(out_dir, f"{cfg['run_name']}_state.pt")
    if cfg["resume"] and os.path.exists(state_path):
        cand = torch.load(state_path, map_location=device, weights_only=False)
        saved = cand.get("arch")
        reason = None
        if saved is not None:
            diff = {k: (saved[k], arch[k]) for k in saved
                    if k in arch and saved[k] != arch[k]}
            if diff:
                reason = f"architecture changed: {diff}"
        prior = cand.get("sched")
        if reason is None and prior is not None:
            sdiff = {k: (prior[k], sched[k]) for k in prior
                     if k in sched and tuple_of(prior[k]) != tuple_of(sched[k])}
            if sdiff:
                reason = f"training schedule changed: {sdiff}"
        if reason is None:
            try:
                raw.load_state_dict(cand["raw"])
            except RuntimeError as exc:
                reason = f"state_dict does not fit: {str(exc).splitlines()[0]}"
        if reason is None:
            state = cand
            log(f"[resume] {os.path.basename(state_path)} at epoch "
                f"{state['epoch']}", rank=rank)
        elif rank == 0:
            stale = state_path.replace(".pt", "_stale.pt")
            os.replace(state_path, stale)
            log(f"[resume] ignoring {os.path.basename(state_path)}: {reason}",
                rank=rank)
            log(f"[resume] kept as {os.path.basename(stale)}; starting fresh",
                rank=rank)
        del cand

    best_path = os.path.join(out_dir, f"{cfg['run_name']}_best.pt")
    if not state and rank == 0 and os.path.exists(best_path):
        prev = best_path.replace(".pt", "_prev.pt")
        os.replace(best_path, prev)
        log(f"[resume] fresh start: kept the existing checkpoint as "
            f"{os.path.basename(prev)} so this run cannot overwrite it with "
            f"an early-epoch model", rank=rank)

    model = raw
    if world > 1:
        model = nn.parallel.DistributedDataParallel(
            raw, device_ids=[local] if device.type == "cuda" else None)

    ema = EMA(raw, cfg["ema_decay"])
    if state.get("ema"):
        ema.shadow = {k: v.to(device) for k, v in state["ema"].items()}
    gen = torch.Generator(device=device).manual_seed(cfg["seed"] + rank)

    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"],
                            weight_decay=cfg["weight_decay"], betas=(0.9, 0.95))
    if state.get("opt"):
        opt.load_state_dict(state["opt"])

    start_epoch = int(state.get("epoch", 0)) + 1
    best = float(state.get("best", -1e9))
    best_epoch = int(state.get("best_epoch", 0))
    best_survive = float(state.get("best_survive", 0.0))
    stale_vals = int(state.get("stale_vals", 0))
    prev_stage = int(state.get("stage", -1))
    stop_on = int(cfg["patience"]) > 0
    if not stop_on:
        log("[setup] early stopping disabled (patience=0); the no-gain counter "
            "is still logged, so the epoch any patience would have fired at "
            "can be read straight off the log", rank=rank)

    if start_epoch > cfg["epochs"]:
        log(f"[resume] state is already at epoch {start_epoch - 1} of "
            f"{cfg['epochs']}: nothing left to train. Delete "
            f"{os.path.basename(state_path)} or raise epochs to train more.",
            rank=rank)

    batch = max(2, cfg["batch"] // world)
    t_start = time.time()
    n_bad = 0
    stopped = ""

    for epoch in range(start_epoch, cfg["epochs"] + 1):
        unroll, stage = unroll_for(epoch, cfg)
        if stage != prev_stage:
            if prev_stage >= 0:
                log(f"[stage] unroll -> {unroll}; max trainable dt "
                    f"{train_s.max_dt(unroll):.1f}", rank=rank)
            prev_stage = stage
        bs = max(2, batch // max(1, min(unroll, cfg["grad_steps"])))
        frac = (epoch - 1) / max(1, cfg["epochs"] - 1)
        cos = 0.5 * (1 + math.cos(math.pi * frac))
        lr = (cfg["lr_min"] + (cfg["lr"] - cfg["lr_min"]) * cos) \
            * cfg["stage_lr_mult"] ** stage
        for g in opt.param_groups:
            g["lr"] = lr

        t_epoch = time.time()
        run_loss, run_dpsi, n_seen = 0.0, 0.0, 0
        for _ in range(cfg["steps_per_epoch"]):
            chain, dt_t = train_s.sample(bs, unroll)
            chain = train_s.augment(chain, gen)
            sigma = cfg["noise"] * float(torch.rand((), generator=gen,
                                                    device=device))
            psi = add_noise(chain[0], sigma)

            n_nograd = max(0, unroll - cfg["grad_steps"])
            if n_nograd:
                with torch.no_grad():
                    for _ in range(n_nograd):
                        psi = raw(psi, dt_t)
                psi = psi.detach()

            terms, dpsi = [], 0.0
            for k in range(n_nograd, unroll):
                psi_in = psi
                pred, true = model(psi, dt_t), chain[k + 1]
                terms.append(r2_loss(pred, true)
                             + cfg["w_inc"] * increment_loss(pred, true,
                                                             psi.detach())
                             + cfg["w_grad"] * gradient_loss(pred, true))
                if k == n_nograd:
                    with torch.no_grad():
                        ref = psi_in - psi_in.mean(dim=(1, 2, 3), keepdim=True)
                        dpsi = float((_sq(pred - psi_in)
                                      / _sq(ref).clamp_min(1e-12)).sqrt().mean())
                psi = pred
            loss = torch.stack(terms).mean()

            opt.zero_grad(set_to_none=True)
            if not torch.isfinite(loss):
                n_bad += 1
                continue
            loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(model.parameters(),
                                                cfg["grad_clip"])
            if not torch.isfinite(gn):
                n_bad += 1
                opt.zero_grad(set_to_none=True)
                continue
            opt.step()
            ema.update(raw)

            run_loss += float(loss.detach())
            run_dpsi += dpsi
            n_seen += 1

        dt_epoch = time.time() - t_epoch
        ms = dt_epoch / max(1, cfg["steps_per_epoch"]) * 1e3
        line = (f"epoch {epoch:3d}/{cfg['epochs']} | unroll {unroll:2d} | "
                f"train {run_loss / max(1, n_seen):.5f} | ")

        do_val = (epoch % cfg["val_every"] == 0) or epoch == cfg["epochs"]
        if rank == 0:
            evm = UNO(**arch).to(device, dtype=dtype)
            evm.load_state_dict(raw.state_dict())
            ema.copy_to(evm)
            val1 = eval_one_step(evm, val_s)
            line += (f"val1 {val1:.5f} | d/psi {run_dpsi / max(1, n_seen):.4f} "
                     f"| lr {lr:.2e} | {dt_epoch:.1f}s ({ms:.0f} ms/step)")
            if do_val:
                score, surv, per_seed = eval_rollout(evm, gt_paths, cfg,
                                                     device, dtype)
                line += (f" | rollout {score:.4f} | survives t={surv:7.0f} "
                         f"{[int(s) for s in per_seed]}")
                if score > best + cfg["min_delta"]:
                    best, best_epoch, best_survive = score, epoch, surv
                    stale_vals = 0
                    torch.save(dict(model=evm.state_dict(), arch=arch,
                                    precision=prec,
                                    metrics=dict(rollout=score, survives=surv,
                                                 per_seed=per_seed,
                                                 epoch=epoch, val1=val1),
                                    config=cfg),
                               best_path)
                    line += "  <- best"
                else:
                    stale_vals += 1
                    cap = cfg["patience"] if stop_on else "off"
                    line += f"  (no gain {stale_vals}/{cap})"
                torch.save(dict(epoch=epoch, best=best, best_epoch=best_epoch,
                                best_survive=best_survive,
                                stale_vals=stale_vals, stage=stage,
                                arch=arch, sched=sched,
                                raw=raw.state_dict(), ema=ema.shadow,
                                opt=opt.state_dict()), state_path)
            del evm
            log(line)
        if world > 1:
            dist.barrier()

        if do_val and stop_on and stale_vals >= cfg["patience"]:
            stopped = " [stopped early: no further improvement]"
            log(f"[stop] rollout score has not improved by {cfg['min_delta']} "
                f"for {cfg['patience']} validations "
                f"({cfg['patience'] * cfg['val_every']} epochs); "
                f"best was "
                f"{best:.4f} at epoch {best_epoch}", rank=rank)
            break

    total = time.time() - t_start
    if rank == 0:
        if device.type == "cuda":
            log(f"[mem] peak allocated "
                f"{torch.cuda.max_memory_allocated() / 2**30:.2f} GB")
        if n_bad:
            log(f"[warn] skipped {n_bad} non-finite steps")
        log(f"[done] best rollout {best:.4f} (survives to t = "
            f"{best_survive:.0f}, epoch {best_epoch}) | "
            f"total {total / 3600:.2f} h ({total:.0f}s) | "
            f"{n_par:,} parameters | saved to "
            f"{best_path}{stopped}")
        with open(os.path.join(out_dir,
                               f"{cfg['run_name']}_train_config.json"),
                  "w") as f:
            json.dump({k: (list(v) if isinstance(v, tuple) else v)
                       for k, v in cfg.items()}, f, indent=2)


def main():
    cfg: dict[str, Any] = dict(CONFIG)
    smoke = "--smoke" in sys.argv
    if smoke:
        cfg.update(SMOKE)

    rank, world, local, device = init_distributed()
    out_dir = setup_output(cfg, rank, "train")
    prec, dtype = set_precision()

    if "--bench" in sys.argv:
        benchmark(cfg, device, dtype)
        return

    log("=" * 78, rank=rank)
    log(f" UNO / Cahn-Hilliard -- TRAINING{'  [SMOKE]' if smoke else ''}   "
        f"{datetime.now():%Y-%m-%d %H:%M:%S}", rank=rank)
    log("=" * 78, rank=rank)
    train(cfg, device, dtype, prec, rank, world, local, out_dir, smoke)
    if world > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
