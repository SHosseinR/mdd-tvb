"""M5.4: Whittle-likelihood fitting of the stable-regime dual Jansen-Rit network.

Model (same physics as the frozen M5.3): analytic linear-regime cross-spectrum
of the delayed dual Jansen-Rit network, a shared delayed alpha drive, network x
hemisphere input-noise gains, a lead-field-projected aperiodic source
background and diagonal observation noise.

Changes relative to M5.3:
* objective: complex-Wishart (Whittle) likelihood of the full cross-spectrum in
  observed coordinates (``mdd_tvb.whittle``) instead of reliability-selected
  features; the subject's overall scale is profiled out; channels prone to
  scalp muscle activity are marginalised above 20 Hz;
* parameters: conduction speed, fast-generator ratio and fraction and the
  noise time constant are fixed at the population value; limbic and control
  gains are tied across hemispheres (none of these was recoverable, audit
  section 5.5);
* inference: MAP with Gaussian priors, then a Laplace posterior from the
  expected (Fisher) information of the Whittle likelihood.
"""
from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
from mdd_tvb import linear_jax as LJ  # noqa: E402
from mdd_tvb import whittle as W  # noqa: E402
from mdd_tvb.spectral_features import load_cross_spectral_collection  # noqa: E402

FIXED_GLOBALS = ("speed_mm_per_ms", "fast_ratio", "fast_fraction", "noise_tau_ms")
FREE_GLOBALS = tuple(n for n in LJ.GLOBAL_NAMES if n not in FIXED_GLOBALS)
FREE_INDEX = np.asarray([LJ.GLOBAL_NAMES.index(n) for n in FREE_GLOBALS])
TIED = ("Cont", "Limbic")
NUISANCE = ("obs_fraction", "obs_exponent", "src_fraction", "src_exponent", "common_share")
COMMON_ORIGIN = (0.0, -15.0, 60.0)
COMMON_SPEED = 3.0


def spatial_names(group_names):
    names = []
    for g in group_names:
        net = g.rsplit("_", 1)[0]
        name = net if net in TIED else g
        if name not in names:
            names.append(name)
    return names


def spatial_map(group_names):
    names = spatial_names(group_names)
    M = np.zeros((len(group_names), len(names)))
    for i, g in enumerate(group_names):
        net = g.rsplit("_", 1)[0]
        M[i, names.index(net if net in TIED else g)] = 1.0
    return M, names


@dataclass
class Setup:
    static: object
    lead: object
    source_cov: object
    mapping: object
    spatial: list
    freq: object
    u_population: np.ndarray  # (10,) unconstrained globals of the population fit
    z_population: np.ndarray  # (d + 5,)

    @property
    def d(self):
        return self.mapping.shape[1]

    @property
    def n_theta(self):
        return len(FREE_INDEX) + self.d + len(NUISANCE)

    def theta_names(self):
        return list(FREE_GLOBALS) + [f"log_gain_{n}" for n in self.spatial] + list(NUISANCE)


def make_setup(lead_name: str, population: dict | None = None) -> Setup:
    from mdd_tvb.spectral_config import load_spectral_m5_config
    from mdd_tvb.linear_spectral import CandidateLinearizer

    cfg = load_spectral_m5_config(ROOT / "configs/m5_spectral.toml")
    lin = CandidateLinearizer(cfg)
    static = LJ.static_from_linearizer(lin, cfg)
    if lead_name == "bem":
        from mdd_tvb.template_bem import load_template_bem_gain
        gain, _, _ = load_template_bem_gain(ROOT / "data/forward/template_bem_corrected/template_bem_schaefer200_gain.npz")
    else:
        gain = lin.gain
    lead = LJ.average_reference(gain)
    g = np.asarray(gain) - np.asarray(gain).mean(0, keepdims=True)
    cov = g @ g.T
    mapping, names = spatial_map(list(static.group_names))
    freq = np.arange(cfg.spectral.frequency_min_hz, cfg.spectral.frequency_max_hz + 0.5, cfg.spectral.frequency_bin_hz)
    if population is None:
        u_pop = np.zeros(10)
        z_pop = np.concatenate([np.zeros(len(names)), [0.0, 0.0, -1.0, 0.0, -1.5]])
    else:
        u_pop = np.asarray(population["u"], float)
        z_pop = np.asarray(population["z"], float)
    return Setup(static, lead, jnp.asarray(cov / np.mean(np.diag(cov))), jnp.asarray(mapping), names,
                 jnp.asarray(freq), u_pop, z_pop)


def full_u(setup: Setup, u_free):
    u = jnp.asarray(setup.u_population)
    return u.at[FREE_INDEX].set(u_free)


def combine(setup: Setup, C, common, z):
    """Unscaled sensor CSD from unit-gain group contributions (same algebra as M5.3)."""
    d = setup.d
    beta = setup.mapping @ (z[:d] - jnp.mean(z[:d]))
    frac = 0.95 * jax.nn.sigmoid(z[d])
    expo = 2.5 * jax.nn.sigmoid(z[d + 1])
    sfrac = 0.95 * jax.nn.sigmoid(z[d + 2])
    sexp = 3.0 * jax.nn.sigmoid(z[d + 3])
    share = 0.95 * jax.nn.sigmoid(z[d + 4])
    freq = setup.freq
    csd = jnp.einsum("k,kfcd->fcd", jnp.exp(beta), C)
    level = jnp.mean(jnp.real(jnp.diagonal(csd, axis1=1, axis2=2)))
    cdiag = jnp.mean(jnp.real(jnp.diagonal(common, axis1=1, axis2=2)))
    csd = csd + common * (level / cdiag) * share / (1.0 - share)
    level = jnp.mean(jnp.real(jnp.diagonal(csd, axis1=1, axis2=2)))
    shape = freq ** (-expo)
    shape = shape / jnp.mean(shape)
    out = csd + (level * frac / (1.0 - frac) * shape)[:, None, None] * jnp.eye(26)[None]
    sshape = freq ** (-sexp)
    sshape = sshape / jnp.mean(sshape)
    out = out + (level * sfrac / (1.0 - sfrac) * sshape)[:, None, None] * setup.source_cov[None]
    return out


def contributions(setup: Setup, u10):
    phys = LJ.unit_to_physical(u10)
    p, speed = LJ.build_state(phys, setup.static)
    C, common, psi = LJ.contributions_with_common_drive(
        p, setup.static, setup.lead, LJ.delays_for_speed(setup.static, speed), jnp.asarray(COMMON_ORIGIN), COMMON_SPEED)
    return C, common, p, psi


def model_csd(setup: Setup, theta):
    k = len(FREE_INDEX)
    C, common, p, psi = contributions(setup, full_u(setup, theta[:k]))
    return combine(setup, C, common, theta[k:]), p, psi


def stability(setup: Setup, p, psi):
    absc = LJ.node_abscissa_per_s(p, psi)
    margin = LJ.fold_margin(p, psi)
    sg = LJ.small_gain(p, psi, jnp.asarray(setup.static.fine_frequency_hz))
    penalty = (jnp.mean(jax.nn.relu(absc + 2.0) ** 2) * 1e-2 + jnp.mean(jax.nn.relu(0.4 - margin) ** 2) * 10.0
               + jax.nn.relu(sg - 0.9) ** 2 * 10.0)
    certified = (jnp.max(absc) < -1.0) & (jnp.min(margin) > 0.3) & (sg < 1.0)
    return penalty, certified


@dataclass
class Prior:
    mean: np.ndarray
    sd: np.ndarray

    def cost(self, theta):
        return 0.5 * jnp.sum(((theta - self.mean) / self.sd) ** 2)


def default_prior(setup: Setup) -> Prior:
    k = len(FREE_INDEX)
    mean = np.concatenate([setup.u_population[FREE_INDEX], setup.z_population])
    sd = np.concatenate([np.full(k, 1.0), np.full(setup.d, 1.0), np.full(len(NUISANCE), 1.5)])
    return Prior(mean, sd)


# ----------------------------------------------------------------------------
# data
# ----------------------------------------------------------------------------
@dataclass
class Half:
    Cc: np.ndarray    # (N, F, 25, 25) complex, observed coordinates, unit mean power
    scale: np.ndarray  # (N,) normalisation constants (raw = Cc * scale)
    nu: np.ndarray    # (N,) Wishart degrees of freedom


@dataclass
class SubjectData:
    ids: list
    groups: list
    D: np.ndarray     # (N, F, 25, 26)
    A: np.ndarray     # (N, F, 25, 26)
    pad: np.ndarray   # (N, F, 25)
    fit: Half
    val: Half
    raw_fit: np.ndarray  # (N, F, 26, 26) raw CSDs (for feature-based metrics)
    raw_val: np.ndarray


def load_subjects(emp_dir, ids=None, use_emg_flags=True, fixed_emg=W.FIXED_EMG_CHANNELS) -> SubjectData:
    emp = ROOT / emp_dir
    fit = load_cross_spectral_collection(emp / "cross_spectra_fit.npz")
    val = load_cross_spectral_collection(emp / "cross_spectra_validation.npz")
    labels = list(fit.channel_names.astype(str))
    order = list(fit.subject_ids.astype(str))
    ids = order if ids is None else [s for s in ids if s in set(order)]
    idx = [order.index(s) for s in ids]
    qc = pd.read_csv(emp / "qc.csv").set_index("subject_id") if (emp / "qc.csv").is_file() else None
    repair = {}
    if (emp / "repair_matrices.npz").is_file():
        with np.load(emp / "repair_matrices.npz") as payload:
            repair = dict(zip(payload["subject_ids"].astype(str), payload["repair"].astype(float)))
    D, A, pad, halves = [], [], [], {"fit": ([], [], []), "val": ([], [], [])}
    for s, i in zip(ids, idx):
        emg = ()
        if use_emg_flags and qc is not None and isinstance(qc.loc[s, "emg_channels"], str):
            emg = tuple(qc.loc[s, "emg_channels"].split(";"))
        d, a, pd_ = W.observation_operators(labels, fit.frequency_hz, emg, repair.get(s), fixed_emg=fixed_emg)
        D.append(d), A.append(a), pad.append(pd_)
        for key, coll in (("fit", fit), ("val", val)):
            Cc, scale = W.project_data(coll.csd[i], d, pd_)
            halves[key][0].append(Cc)
            halves[key][1].append(scale)
            halves[key][2].append(W.DOF_PER_EPOCH * coll.epoch_counts[i])
    mk = lambda k: Half(np.stack(halves[k][0]), np.asarray(halves[k][1]), np.asarray(halves[k][2], float))  # noqa: E731
    return SubjectData(ids, [str(fit.groups[i]) for i in idx], np.stack(D), np.stack(A), np.stack(pad),
                       mk("fit"), mk("val"), fit.csd[idx], val.csd[idx])


def pooled_null(raw_csds):
    """Population CSD: mean of per-subject power-normalised CSDs."""
    norm = [c / np.mean(np.real(np.einsum("fii->fi", c))) for c in raw_csds]
    return np.mean(norm, axis=0)


def x64() -> bool:
    return os.environ.get("MDD_TVB_JAX_X64", "1") != "0"


def save_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)))
