"""Distributed fsaverage BEM observation model aggregated to Schaefer parcels."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import mne
import numpy as np
import pandas as pd


def sha256(path: Path, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def tdbrain_table_positions(
    coordinate_file: Path, channel_names: tuple[str, ...]
) -> np.ndarray:
    table = pd.read_csv(coordinate_file).set_index("label")
    missing = [name for name in channel_names if name not in table.index]
    if missing:
        raise ValueError(f"TDBRAIN coordinate table is missing channels: {missing}")
    positions_mm = table.loc[list(channel_names), ["x_mm", "y_mm", "z_mm"]].to_numpy(
        dtype=float
    )
    if not np.isfinite(positions_mm).all():
        raise ValueError("TDBRAIN coordinates contain non-finite values")
    return positions_mm / 1000.0


def tdbrain_montage(
    coordinate_file: Path, channel_names: tuple[str, ...]
) -> mne.channels.DigMontage:
    """Map published TDBRAIN points in Colin27/fsaverage MRI coordinates.

    The table positions closely match MNE's ``colin27_1005`` MRI-coordinate
    sensor positions. Labeling them as *head* coordinates skips the fiducial
    transform and displaces some sensors by centimetres on fsaverage.
    """

    positions = tdbrain_table_positions(coordinate_file, channel_names)
    standard = mne.channels.make_standard_montage("colin27_1005")
    standard_positions = standard.get_positions()
    if standard_positions["coord_frame"] != "mri":
        raise RuntimeError("Colin27 montage is not in the expected MRI frame")
    return mne.channels.make_dig_montage(
        ch_pos=dict(zip(channel_names, positions, strict=True)),
        nasion=standard_positions["nasion"],
        lpa=standard_positions["lpa"],
        rpa=standard_positions["rpa"],
        coord_frame="mri",
    )


def _vertex_areas(source_space: dict[str, Any]) -> np.ndarray:
    rr = np.asarray(source_space["rr"], dtype=float)
    triangles = np.asarray(source_space["tris"], dtype=int)
    triangle_area = 0.5 * np.linalg.norm(
        np.cross(
            rr[triangles[:, 1]] - rr[triangles[:, 0]],
            rr[triangles[:, 2]] - rr[triangles[:, 0]],
        ),
        axis=1,
    )
    area = np.zeros(len(rr), dtype=float)
    for corner in range(3):
        np.add.at(area, triangles[:, corner], triangle_area / 3.0)
    return area


def aggregate_fixed_forward_to_labels(
    fixed_gain: np.ndarray,
    source_spaces: mne.SourceSpaces,
    labels: list[mne.Label],
    expected_names: list[str],
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Area-weight a fixed-normal vertex gain into equal-total parcel columns."""

    if fixed_gain.shape[1] != sum(int(space["nuse"]) for space in source_spaces):
        raise ValueError("Fixed gain columns do not match active source vertices")
    labels_by_name = {
        label.name.removesuffix("-lh").removesuffix("-rh"): label
        for label in labels
        if "Medial_Wall" not in label.name
    }
    missing = [name for name in expected_names if name not in labels_by_name]
    extra = sorted(set(labels_by_name).difference(expected_names))
    if missing or extra:
        raise ValueError(
            f"Schaefer annotation/order mismatch; missing={missing}, extra={extra}"
        )

    offsets = np.cumsum([0] + [int(space["nuse"]) for space in source_spaces])
    area = [_vertex_areas(space) for space in source_spaces]
    reference = np.eye(fixed_gain.shape[0]) - np.ones(
        (fixed_gain.shape[0], fixed_gain.shape[0])
    ) / fixed_gain.shape[0]
    referenced_vertex_gain = reference @ fixed_gain
    regional = np.empty((fixed_gain.shape[0], len(expected_names)), dtype=float)
    audit: list[dict[str, Any]] = []
    for region_index, name in enumerate(expected_names):
        label = labels_by_name[name]
        hemi_index = 0 if label.hemi == "lh" else 1
        active_vertices = np.asarray(source_spaces[hemi_index]["vertno"], dtype=int)
        vertices, active_indices, _ = np.intersect1d(
            active_vertices, label.vertices, return_indices=True
        )
        if not len(vertices):
            raise ValueError(f"No active template sources found for {name}")
        weights = area[hemi_index][vertices]
        weights /= weights.sum()
        columns = offsets[hemi_index] + active_indices
        regional[:, region_index] = fixed_gain[:, columns] @ weights
        vertex_fields = referenced_vertex_gain[:, columns]
        mean_vertex_norm = float(np.linalg.norm(vertex_fields, axis=0) @ weights)
        regional_norm = float(np.linalg.norm(vertex_fields @ weights))
        audit.append(
            {
                "region_index": region_index,
                "region_name": name,
                "hemisphere": label.hemi,
                "source_vertices": int(len(vertices)),
                "represented_area_m2": float(area[hemi_index][vertices].sum()),
                "mean_vertex_field_norm": mean_vertex_norm,
                "regional_field_norm": regional_norm,
                "signed_field_retention": regional_norm
                / max(mean_vertex_norm, np.finfo(float).tiny),
            }
        )
    return regional, audit


def read_schaefer_labels(annotation_dir: Path) -> list[mne.Label]:
    labels: list[mne.Label] = []
    for hemisphere in ("lh", "rh"):
        path = (
            annotation_dir
            / f"{hemisphere}.Schaefer2018_200Parcels_7Networks_order.annot"
        )
        labels.extend(
            mne.read_labels_from_annot(
                "fsaverage",
                hemi=hemisphere,
                annot_fname=path,
                sort=False,
                verbose=False,
            )
        )
    return [label for label in labels if "Medial_Wall" not in label.name]


def build_template_bem_gain(
    *,
    subjects_dir: Path,
    coordinate_file: Path,
    atlas_file: Path,
    annotation_dir: Path,
    channel_names: tuple[str, ...],
    n_jobs: int = 1,
) -> tuple[np.ndarray, dict[str, Any], pd.DataFrame]:
    fsaverage = subjects_dir / "fsaverage"
    bem_dir = fsaverage / "bem"
    source_path = bem_dir / "fsaverage-ico-5-src.fif"
    bem_path = bem_dir / "fsaverage-5120-5120-5120-bem-sol.fif"
    trans_path = bem_dir / "fsaverage-trans.fif"
    for path in (source_path, bem_path, trans_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    info = mne.create_info(list(channel_names), sfreq=500.0, ch_types="eeg")
    info.set_montage(
        tdbrain_montage(coordinate_file, channel_names), on_missing="raise"
    )
    electrode_scalp_distances = mne.dig_mri_distances(
        info,
        str(trans_path),
        "fsaverage",
        subjects_dir=str(subjects_dir),
        dig_kinds="eeg",
        verbose=False,
    )
    if len(electrode_scalp_distances) != len(channel_names):
        raise RuntimeError("Template scalp audit omitted EEG electrodes")
    if float(np.max(electrode_scalp_distances)) > 0.010:
        raise ValueError(
            "TDBRAIN electrodes are >10 mm from the fsaverage scalp; "
            "check the MRI-to-head coordinate transform"
        )
    forward = mne.make_forward_solution(
        info,
        trans=str(trans_path),
        src=str(source_path),
        bem=str(bem_path),
        meg=False,
        eeg=True,
        mindist=5.0,
        n_jobs=n_jobs,
        on_inside="raise",
        verbose=True,
    )
    fixed = mne.convert_forward_solution(
        forward,
        surf_ori=True,
        force_fixed=True,
        use_cps=True,
        copy=True,
        verbose=False,
    )
    vertex_gain = np.asarray(fixed["sol"]["data"], dtype=float)
    labels = read_schaefer_labels(annotation_dir)
    atlas = pd.read_csv(atlas_file)
    expected_names = atlas["ROI Name"].astype(str).tolist()
    regional_gain, region_audit = aggregate_fixed_forward_to_labels(
        vertex_gain, fixed["src"], labels, expected_names
    )

    average_reference = np.eye(len(channel_names)) - np.ones(
        (len(channel_names), len(channel_names))
    ) / len(channel_names)
    regional_gain = average_reference @ regional_gain
    raw_rms = float(np.sqrt(np.mean(regional_gain**2)))
    if not np.isfinite(raw_rms) or raw_rms <= 0:
        raise RuntimeError("Template BEM gain has invalid scale")
    normalized_gain = regional_gain / raw_rms

    standard = mne.channels.make_standard_montage("colin27_1005")
    standard_ch = standard.get_positions()["ch_pos"]
    table_positions = tdbrain_table_positions(coordinate_file, channel_names)
    standard_positions = np.asarray([standard_ch[name] for name in channel_names])
    table_unit = table_positions / np.linalg.norm(
        table_positions, axis=1, keepdims=True
    )
    standard_unit = standard_positions / np.linalg.norm(
        standard_positions, axis=1, keepdims=True
    )
    angular_difference = np.degrees(
        np.arccos(np.clip(np.sum(table_unit * standard_unit, axis=1), -1.0, 1.0))
    )
    region_table = pd.DataFrame(region_audit)
    metadata: dict[str, Any] = {
        "method": "MNE three-layer fsaverage BEM with fixed cortical-normal sources",
        "electrode_coordinate_frame": "Colin27/fsaverage MRI, transformed to head by MNE fiducials",
        "electrode_to_scalp_distance_mm": {
            "median": float(np.median(electrode_scalp_distances) * 1000.0),
            "maximum": float(np.max(electrode_scalp_distances) * 1000.0),
        },
        "mne_version": mne.__version__,
        "subject": "fsaverage",
        "source_space": source_path.name,
        "bem_solution": bem_path.name,
        "transform": trans_path.name,
        "conductivity_s_per_m": [0.3, 0.006, 0.3],
        "aggregation": (
            "area-weighted mean of fixed-normal vertex lead fields within each "
            "Schaefer parcel; every parcel has equal total source amplitude"
        ),
        "reference": "average",
        "normalization": "whole-gain RMS set to one; absolute EEG gain is nuisance",
        "gain_shape": list(normalized_gain.shape),
        "vertex_gain_shape": list(vertex_gain.shape),
        "raw_gain_rms": raw_rms,
        "rank_after_average_reference": int(np.linalg.matrix_rank(normalized_gain)),
        "minimum_vertices_per_parcel": int(region_table.source_vertices.min()),
        "median_vertices_per_parcel": float(region_table.source_vertices.median()),
        "maximum_vertices_per_parcel": int(region_table.source_vertices.max()),
        "signed_field_retention": {
            "median": float(region_table.signed_field_retention.median()),
            "p05": float(region_table.signed_field_retention.quantile(0.05)),
            "minimum": float(region_table.signed_field_retention.min()),
        },
        "tdbrain_to_colin27_angular_difference_degrees": {
            "median": float(np.median(angular_difference)),
            "maximum": float(np.max(angular_difference)),
        },
        "hashes": {
            "coordinate_file": sha256(coordinate_file),
            "atlas_file": sha256(atlas_file),
            "lh_annotation": sha256(
                annotation_dir
                / "lh.Schaefer2018_200Parcels_7Networks_order.annot"
            ),
            "rh_annotation": sha256(
                annotation_dir
                / "rh.Schaefer2018_200Parcels_7Networks_order.annot"
            ),
            "source_space": sha256(source_path),
            "bem_solution": sha256(bem_path),
            "transform": sha256(trans_path),
        },
        "limitations": [
            "Template anatomy, not subject-specific anatomy.",
            "Published electrode coordinates, not individual digitization.",
            "Regional neural states are represented as uniform parcel source amplitudes.",
            "Absolute gain is not calibrated to EEG microvolts.",
        ],
    }
    return normalized_gain, metadata, region_table


def save_template_bem_gain(
    output_dir: Path,
    gain: np.ndarray,
    channel_names: tuple[str, ...],
    region_names: list[str],
    metadata: dict[str, Any],
    region_audit: pd.DataFrame,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_dir / "template_bem_schaefer200_gain.npz",
        gain=gain,
        channel_names=np.asarray(channel_names, dtype="U16"),
        region_names=np.asarray(region_names, dtype="U96"),
    )
    (output_dir / "template_bem_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    region_audit.to_csv(output_dir / "template_bem_region_audit.csv", index=False)


def load_template_bem_gain(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with np.load(path) as payload:
        gain = np.asarray(payload["gain"], dtype=float)
        channel_names = np.asarray(payload["channel_names"])
        region_names = np.asarray(payload["region_names"])
    if gain.shape != (len(channel_names), len(region_names)):
        raise ValueError("Template BEM gain axes are inconsistent")
    if not np.isfinite(gain).all():
        raise ValueError("Template BEM gain contains non-finite values")
    return gain, channel_names, region_names
