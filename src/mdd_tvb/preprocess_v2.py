"""Artefact-aware TDBRAIN preprocessing (v2) from the raw BDF files.

Re-implements the TDBRAIN authors' automatic pipeline (van Dijk et al. 2022,
github.com/brainclinics/TDBRAIN, ``autopreprocessing.py``) with their default
thresholds, in the same order:

1. bipolar EOG (VEOG = VPVA - VNVB, HEOG = HPHL - HNHR), demean;
2. zero-phase 50 Hz notch (IIR, Q = 100), 0.5 Hz high-pass and 100 Hz low-pass
   (4th-order Butterworth);
3. Gratton-type EOG regression;
4. detection of EMG (75-95 Hz envelope, z > 4 and > 3 uV), jumps (median-filtered
   derivative, z > 5 and > 30 uV), kurtosis (> 8 in 4-s windows), extreme voltage
   swings (> 200 uV peak-to-peak in 0.5-s windows) and residual eye blinks
   (0.5-6 Hz envelope, z > 0.5 and > 60 uV);
5. bad channels (artefacts in > 1/3 of the recording, broadband 55-95 Hz outliers,
   bridging) repaired from the authors' neighbour table with their
   inverse-distance weights; the recording fails QC if more than 3 channels need
   repair or more than 2/3 of the samples are artefactual.

Two documented deviations:

* EOG regression uses one set of propagation coefficients per recording,
  estimated and applied on EOG/EEG low-passed at 7 Hz (the authors use 15 Hz
  within detected eye-movement windows).  Eyes-closed ocular artefacts are slow;
  a 7-Hz limit keeps frontal alpha that leaks into the EOG electrodes from being
  subtracted from Fp1/Fp2/F7/F8.
* The broadband-outlier rule uses a robust z-score (median/MAD, > 3.5) across
  channels instead of a plain z > 1.96, which flags about one normal channel per
  recording when there are 26 channels.

Finally the data are average referenced and only 4-s Welch epochs that lie
entirely inside artefact-free time are used for the spectra.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.signal import butter, filtfilt, hilbert, iirnotch, medfilt, sosfiltfilt
from scipy.signal.windows import hann
from scipy.stats import kurtosis

EEG_LABELS = ("Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8", "FC3", "FCz", "FC4", "T7", "C3", "Cz",
              "C4", "T8", "CP3", "CPz", "CP4", "P7", "P3", "Pz", "P4", "P8", "O1", "Oz", "O2")
EOG_LABELS = ("VPVA", "VNVB", "HPHL", "HNHR")

# Authors' neighbour table (self-reference of F7 removed; T7 kept as in the original).
NEIGHBOURS = {
    "Fp1": ["Fp2", "F7", "F3"], "Fp2": ["Fp1", "F8", "F4"], "F7": ["Fp1", "F3"],
    "F3": ["Fp1", "Fz", "FC3", "F7"], "Fz": ["F4", "FCz", "F3"], "F4": ["Fp2", "F8", "FC4", "Fz"],
    "F8": ["Fp2", "F4", "T8"], "FC3": ["F3", "C3", "FCz"], "FCz": ["Fz", "FC3", "FC4", "Cz"],
    "FC4": ["F4", "FCz", "C4"], "T7": ["F7", "P7", "C3"], "C3": ["FC3", "Cz", "CP3"],
    "Cz": ["FCz", "CPz", "C3", "C4"], "C4": ["Cz", "CP4", "FC4"], "T8": ["F8", "P8", "C4"],
    "CP3": ["C3", "CPz", "P3"], "CPz": ["Cz", "CP4", "CP3", "Pz"], "CP4": ["C4", "P4", "CPz"],
    "P7": ["F7", "P3", "O1"], "P3": ["P7", "CP3", "Pz", "O1"], "Pz": ["P3", "CPz", "P4", "Oz"],
    "P4": ["Pz", "CP4", "P8", "O2"], "P8": ["T8", "P4", "O2"], "O1": ["P7", "P3", "Oz"],
    "Oz": ["O1", "Pz", "O2"], "O2": ["Oz", "P4", "P8"],
}
# Authors' channel coordinates (mm), same order as EEG_LABELS.
COORDS_MM = np.array([
    [84.06, -26.81, -10.56], [83.74, 29.41, -10.04], [41.69, -66.99, -15.96], [51.87, -48.05, 39.87],
    [57.01, 0.9, 66.36], [51.84, 50.38, 41.33], [41.16, 68.71, -15.31], [21.02, -58.83, 54.82],
    [24.63, 0.57, 87.63], [21.16, 60.29, 55.58], [-16.52, -83.36, -12.65], [-13.25, -65.57, 64.98],
    [-11.28, 0.23, 99.81], [-12.8, 66.5, 65.11], [-16.65, 84.44, -11.79], [-48.48, -65.51, 68.57],
    [-48.77, -0.42, 98.37], [-48.35, 65.03, 68.57], [-75.17, -71.46, -3.7], [-80.11, -55.07, 59.44],
    [-82.23, -0.87, 82.43], [-80.13, 53.51, 59.4], [-75.17, 71.1, -3.69], [-114.52, -28.98, 9.67],
    [-117.79, -1.41, 15.84], [-114.68, 26.89, 9.45]])


@dataclass
class PreprocessResult:
    data_uv: np.ndarray            # (26, n) average-referenced, repaired EEG in microvolt
    clean: np.ndarray              # (n,) True where no artefact was detected in any channel
    fs: float
    repaired_channels: list[str]
    emg_channels: list[str]
    bridged_pairs: list[tuple[str, str]]
    repair_matrix: np.ndarray      # (26, 26): repaired = repair_matrix @ original (before re-referencing)
    eog_coefficients: np.ndarray   # (26, 2) VEOG/HEOG propagation coefficients
    qc_ok: bool
    qc_reason: str
    stats: dict = field(default_factory=dict)


def _butter_filtfilt(x, fs, cutoff, btype):
    b, a = butter(4, np.asarray(cutoff) / (0.5 * fs), btype=btype)
    return filtfilt(b, a, x, axis=-1)


def _bandpass(x, fs, lo, hi):
    sos = butter(4, [lo / (0.5 * fs), hi / (0.5 * fs)], btype="bandpass", output="sos")
    return sosfiltfilt(sos, x, axis=-1)


def _pad_mask(mask, pad):
    """Dilate a boolean (..., n) mask by ``pad`` samples on both sides."""
    if pad <= 0 or not mask.any():
        return mask
    kernel = np.ones(2 * pad + 1)
    out = np.empty_like(mask)
    for r in range(mask.shape[0]):
        out[r] = np.convolve(mask[r].astype(float), kernel, mode="same") > 0
    return out


def _zscore_rows(x):
    sd = x.std(axis=1, keepdims=True)
    sd[sd == 0] = np.inf
    return (x - x.mean(axis=1, keepdims=True)) / sd


def detect_emg(x, fs, threshold=4.0, absolute_uv=3.0, padding_s=0.1):
    env = np.abs(hilbert(_bandpass(x, fs, 75.0, 95.0), axis=-1))
    win = hann(int(0.5 * fs), sym=True)
    smooth = np.stack([np.convolve(e, win, mode="same") for e in env])
    hit = (_zscore_rows(smooth) > threshold) & (env > absolute_uv)
    box = np.ones(int(0.5 * fs))
    hit = np.stack([np.convolve(h.astype(float), box, mode="same") > 0 for h in hit])
    return _pad_mask(hit, int(padding_s * fs))


def detect_jumps(x, fs, threshold=5.0, absolute_uv=30.0, padding_s=0.01):
    filt = np.stack([medfilt(r, kernel_size=9) for r in x])
    diff = np.zeros_like(filt)
    diff[:, 1:] = np.abs(np.diff(filt, axis=1))
    hit = (_zscore_rows(diff) > threshold) & (diff > absolute_uv)
    return _pad_mask(hit, int(padding_s * fs))


def _window_stat(x, fs, winlen_s, step_s, stat):
    n = x.shape[1]
    w, step = int(winlen_s * fs), max(1, int(step_s * fs))
    out = np.zeros_like(x)
    for start in range(0, n - w, step):
        out[:, start:start + w] = stat(x[:, start:start + w])[:, None]
    return out


def detect_kurtosis(x, fs, threshold=8.0, padding_s=0.1):
    k = _window_stat(x, fs, 4.0, 0.1, lambda seg: kurtosis(seg, axis=1, fisher=True))
    return _pad_mask(k > threshold, int(padding_s * fs))


def detect_swing(x, fs, threshold_uv=200.0, padding_s=0.05):
    s = _window_stat(x, fs, 0.5, 0.05, lambda seg: seg.max(axis=1) - seg.min(axis=1))
    return _pad_mask(s > threshold_uv, int(padding_s * fs))


def detect_residual_blinks(x, fs, threshold=0.5, absolute_uv=60.0, padding_s=0.1):
    env = np.abs(hilbert(_bandpass(x, fs, 0.5, 6.0), axis=-1))
    win = hann(int(1.0 * fs), sym=True)
    smooth = np.stack([np.convolve(e, win, mode="same") for e in env])
    hit = (_zscore_rows(smooth) > threshold) & (env > absolute_uv)
    return _pad_mask(hit, int(padding_s * fs))


def correct_eog(eeg, eog, fs, lowpass_hz=7.0, max_coefficient=1.0):
    """Remove ocular activity with one regression per recording on < ``lowpass_hz`` data.

    A bipolar EOG derivation whose propagation coefficient exceeds
    ``max_coefficient`` anywhere (a scalp electrode cannot see more ocular
    potential than the peri-ocular pair) is treated as a faulty EOG electrode and
    dropped; the remaining derivation is refitted.
    """
    eog_lp = _butter_filtfilt(eog, fs, lowpass_hz, "lowpass")
    eeg_lp = _butter_filtfilt(eeg, fs, lowpass_hz, "lowpass")
    use = [0, 1]
    while use:
        coef, *_ = np.linalg.lstsq(eog_lp[use].T, eeg_lp.T, rcond=None)
        worst = np.abs(coef).max(axis=1)
        if worst.max() <= max_coefficient:
            break
        use.pop(int(np.argmax(worst)))
    full = np.zeros((eeg.shape[0], 2))
    if use:
        full[:, use] = coef.T
    return eeg - full @ eog_lp, full


def bridged_pairs(x, labels, relative_threshold=0.02, min_correlation=0.99):
    """Electrical-distance bridging (Tenke & Kayser 2001) among neighbouring channels."""
    pairs = []
    ed = {}
    for a, na in enumerate(labels):
        for nb in NEIGHBOURS[na]:
            b = labels.index(nb)
            if b <= a:
                continue
            d = x[a] - x[b]
            ed[(a, b)] = float(np.var(d))
    if not ed:
        return pairs
    median = np.median(list(ed.values()))
    for (a, b), value in ed.items():
        if value < relative_threshold * median and np.corrcoef(x[a], x[b])[0, 1] > min_correlation:
            pairs.append((labels[a], labels[b]))
    return pairs


def repair_matrix(labels, bad):
    """Authors' neighbour interpolation as a linear operator; None if a channel cannot be repaired."""
    P = np.eye(len(labels))
    for ch in bad:
        i = labels.index(ch)
        good = [labels.index(n) for n in NEIGHBOURS[ch] if n not in bad]
        if len(good) < 2:
            return None
        d = np.linalg.norm(COORDS_MM[good] - COORDS_MM[i], axis=1)
        w = d.sum() - d
        P[i] = 0.0
        P[i, good] = w / w.sum()
    return P


def preprocess_bdf(path: str | Path, max_duration_s: float = 120.0) -> PreprocessResult:
    import mne

    raw = mne.io.read_raw_bdf(str(path), preload=True, verbose="ERROR")
    fs = float(raw.info["sfreq"])
    labels = list(EEG_LABELS)
    missing = [c for c in labels + list(EOG_LABELS) if c not in raw.ch_names]
    if missing:
        raise ValueError(f"missing channels {missing}")
    n = min(raw.n_times, int(round(max_duration_s * fs)))
    x = raw.get_data(picks=labels, stop=n) * 1e6
    e = raw.get_data(picks=list(EOG_LABELS), stop=n) * 1e6
    eog = np.stack([e[0] - e[1], e[2] - e[3]])
    x = x - x.mean(axis=1, keepdims=True)
    eog = eog - eog.mean(axis=1, keepdims=True)
    # filters: notch, high-pass, low-pass (zero phase)
    b, a = iirnotch(50.0, 100.0, fs=fs)
    x, eog = filtfilt(b, a, x, axis=-1), filtfilt(b, a, eog, axis=-1)
    x = _butter_filtfilt(_butter_filtfilt(x, fs, 0.5, "highpass"), fs, 100.0, "lowpass")
    eog = _butter_filtfilt(_butter_filtfilt(eog, fs, 0.5, "highpass"), fs, 100.0, "lowpass")
    x, coef = correct_eog(x, eog, fs)

    masks = {
        "emg": detect_emg(x, fs),
        "jump": detect_jumps(x, fs),
        "kurtosis": detect_kurtosis(x, fs),
        "swing": detect_swing(x, fs),
        "blink": detect_residual_blinks(x, fs),
    }
    art = np.zeros_like(x, dtype=bool)
    for m in masks.values():
        art |= m
    frac = art.mean(axis=1)
    bad = {labels[i] for i in np.flatnonzero(frac > 1.0 / 3.0)}
    # broadband (55-95 Hz) outlier channels: tonic muscle activity or a noisy electrode
    spec = np.abs(np.fft.rfft(x * hann(n, sym=False)[None], axis=1)) ** 2
    f = np.fft.rfftfreq(n, 1.0 / fs)
    hf = np.log(spec[:, (f > 55) & (f < 95)].mean(axis=1))
    mad = 1.4826 * np.median(np.abs(hf - np.median(hf)))
    robust_z = (hf - np.median(hf)) / max(mad, 1e-12)
    # Tonic-muscle channels are kept (repairing Fp1/Fp2/F7/F8 from each other is not
    # meaningful); they are reported so that their high frequencies can be excluded.
    emg_channels = [labels[i] for i in np.flatnonzero(robust_z > 3.5)]
    bridges = bridged_pairs(x, labels)
    for p in bridges:
        bad |= set(p)
    flat = [labels[i] for i in np.flatnonzero(x.std(axis=1) < 0.5)]
    bad |= set(flat)
    bad_list = [c for c in labels if c in bad]
    P = repair_matrix(labels, bad_list) if bad_list else np.eye(len(labels))
    qc_ok, reason = True, ""
    if P is None:
        qc_ok, reason, P = False, "unrepairable channel neighbourhood", np.eye(len(labels))
    x = P @ x
    good_rows = [i for i, c in enumerate(labels) if c not in bad]
    # In a tonic-EMG channel, EMG bursts and sharp motor-unit spikes (which the
    # derivative-based jump detector also catches) are part of that channel's
    # flagged muscle state, whose high frequencies are not scored downstream.
    tonic = [labels.index(c) for c in emg_channels]
    art[tonic] = masks["kurtosis"][tonic] | masks["swing"][tonic] | masks["blink"][tonic]
    clean = ~art[good_rows].any(axis=0)
    if len(bad_list) > 3:
        qc_ok, reason = False, f"{len(bad_list)} channels need repair"
    elif clean.mean() < 1.0 / 3.0:
        qc_ok, reason = False, f"only {clean.mean():.0%} artefact-free"
    x = x - x.mean(axis=0, keepdims=True)  # average reference
    # 20-40 Hz log-log slope per channel (muscle signature) for QC
    sel = (f >= 20) & (f <= 40)
    spec_ref = np.abs(np.fft.rfft(x * hann(n, sym=False)[None], axis=1)) ** 2
    slope = np.polyfit(np.log(f[sel]), np.log(spec_ref[:, sel].T + 1e-30), 1)[0]
    stats = {
        "duration_s": n / fs,
        "clean_fraction": float(clean.mean()),
        **{f"{k}_fraction": float(m[good_rows].any(axis=0).mean()) for k, m in masks.items()},
        "slope_20_40": dict(zip(labels, np.round(slope, 3).tolist())),
        "hf_robust_z": dict(zip(labels, np.round(robust_z, 2).tolist())),
        "flat_channels": flat,
    }
    return PreprocessResult(x.astype(np.float32), clean, fs, bad_list, emg_channels,
                            bridges, P, coef, qc_ok, reason, stats)


def clean_epoch_starts(clean: np.ndarray, nperseg: int, step: int) -> np.ndarray:
    """Starts of half-overlapping Welch epochs packed into each artefact-free span."""
    edges = np.flatnonzero(np.diff(np.concatenate([[0], clean.astype(int), [0]])))
    starts = []
    for begin, end in zip(edges[::2], edges[1::2]):
        starts.extend(range(begin, end - nperseg + 1, step))
    return np.asarray(starts, dtype=int)


def split_halves(starts: np.ndarray, nperseg: int) -> tuple[np.ndarray, np.ndarray]:
    """Split epochs into a first and a second half with no shared samples."""
    if len(starts) < 2:
        return starts[:0], starts[:0]
    boundary = starts[len(starts) // 2]
    first = starts[starts + nperseg <= boundary]
    second = starts[starts >= boundary]
    return first, second
