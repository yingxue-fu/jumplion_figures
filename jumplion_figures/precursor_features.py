"""Vendored copy of precursor_features.py. Override via mechanism.features_function in the
config. One deliberate deviation from the original: ms1/ms2_xics_signal_rank are scaled to
0-100 (not the original's 0-1) -- see the comment at that calculation below."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence, Union

import numpy as np
import pandas as pd

PathLike = Union[str, Path, pd.DataFrame]

EVIDENCE_TIER_ORDER = ["reported", "ms2_coelution", "ms1_peak", "prediction_only"]
EVIDENCE_TIER_LABELS = {
    "reported": "DIA-NN reported",
    "ms2_coelution": "MS2 co-elution",
    "ms1_peak": "MS1 isotope peak",
    "prediction_only": "No qualifying peak",
}



DEFAULT_EVIDENCE_FEATURES = [
    "ms1_xics_signal_rank",
    "ms2_xics_signal_rank",
    "ms1_corr_mean",
    "ms2_corr_mean",
    "ms1_isotope_cosine",
    "ms2_fragment_cosine",
    "ms1_usable_xic_fraction_local",
    "ms2_usable_xic_fraction_local",
]


DEFAULT_STABILITY_FEATURES = [
    "MS1.MS2.Cross.Corr",
    "ms2_fragment_cosine",
    "ms2_target_background_fraction",
    "ms2_xics_signal_rank",
    "ms1_corr_median",
]


@dataclass
class ReliabilityConfig:
    precursor_id_col: str = "Precursor.Id"
    run_col: str = "Run"
    group_col: str = "Group"

    evidence_features: Sequence[str] = field(
        default_factory=lambda: list(DEFAULT_EVIDENCE_FEATURES)
    )
    stability_features: Sequence[str] = field(
        default_factory=lambda: list(DEFAULT_STABILITY_FEATURES)
    )

    # Initial transparent model. These are intentionally simple and should
    # be tuned on held-out DE contrasts rather than repeatedly on one contrast.
    evidence_weight: float = 0.70
    stability_weight: float = 0.30

    # Empirical-CDF normalization against the training/development dataset.
    # Values outside the training range are clipped to [0, 1].
    stability_scale_floor: float = 0.05

    # Filtering parameters.
    reliability_threshold: float = 0.50
    min_group_reliable_fraction: float = 0.50


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def _require(df: pd.DataFrame, columns: Sequence[str], name: str) -> None:
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise KeyError(f"{name} is missing required columns: {missing}")

def _median(s: pd.Series) -> float:
    x = pd.to_numeric(s, errors="coerce").dropna()
    return float(x.median()) if len(x) else np.nan

def _mad(s: pd.Series) -> float:
    x = pd.to_numeric(s, errors="coerce").dropna().to_numpy(dtype=float)
    if len(x) == 0:
        return np.nan
    med = np.median(x)
    return float(np.median(np.abs(x - med)))

def _safe_parse_vector(value):
    """
    Parse semicolon-separated XIC vectors stored as strings.
    Empty, NA, or non-numeric tokens become NaN.
    """
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return np.array([], dtype=float)
    if isinstance(value, (list, tuple, np.ndarray)):
        return np.asarray(value, dtype=float)

    s = str(value).strip()
    if not s or s.lower() in {"nan", "na", "none"}:
        return np.array([], dtype=float)

    out = []
    for token in s.split(";"):
        token = token.strip()
        try:
            out.append(float(token))
        except (TypeError, ValueError):
            out.append(np.nan)
    return np.asarray(out, dtype=float)

def _vector_summary(values, mode="sum"):
    """
    Summarize an ion-level vector in linear intensity space.

    mode='sum' is the default for XIC signal because multiple isotope/fragment
    XICs represent additive measured evidence. 'median' is retained only
    for robust diagnostics.
    """
    x = _safe_parse_vector(values)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return np.nan
    if mode == "sum":
        return float(np.sum(x))
    if mode == "mean":
        return float(np.mean(x))
    if mode == "median":
        return float(np.median(x))
    raise ValueError(f"Unknown summary mode: {mode}")

def _vector_positive_fraction(values, threshold=0.0):
    x = _safe_parse_vector(values)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return np.nan
    return float(np.mean(x > threshold))

def _vector_count(values):
    x = _safe_parse_vector(values)
    return int(np.sum(np.isfinite(x)))

def _parse_vector_corr(values):
    return _safe_parse_vector(values)

def _corr_summary(
    values,
    mode="median",
    threshold=None,
    exclude_zero=True,
):
    """
    Summarize per-isotope/per-fragment XIC correlations.

    Negative correlations are clipped to 0 first (a negative correlation means zero positive
    co-elution evidence, not "anti-evidence" -- same definition everywhere this summary feeds
    into), before exclude_zero gets to decide whether those (now-)zero values are dropped.

    A correlation of exactly 0 is treated as "no usable XIC signal" rather
    than as genuine zero similarity, because the upstream extraction may emit
    0 when an isotope/fragment has no signal.

    Therefore, by default, zero-valued correlations are removed before
    calculating median/mean/max or the fraction above threshold.

    If all channels are zero/non-finite, the summary is NaN, indicating that
    no informative XIC correlation was available.
    """
    x = _parse_vector_corr(values)
    x = x[np.isfinite(x)]
    x = np.clip(x, 0.0, None)

    if exclude_zero:
        x = x[x != 0]

    if len(x) == 0:
        return np.nan

    if mode == "median":
        return float(np.median(x))

    if mode == "mean":
        return float(np.mean(x))

    if mode == "sum":
        return float(np.sum(x))

    if mode == "max":
        return float(np.max(x))

    if mode == "high_fraction":
        if threshold is None:
            raise ValueError("threshold is required for high_fraction.")
        return float(np.mean(x >= threshold))

    raise ValueError(f"Unknown correlation summary mode: {mode}")

def _paired_left_right_summary(left, right, mode="mean"):
    """
    Summarize left/right adjacent background after parsing ion-level vectors.

    The default is mean of the two flank-integrals at the ion level.
    Use median for robust flank pooling.
    """
    l = _safe_parse_vector(left)
    r = _safe_parse_vector(right)

    n = max(len(l), len(r))
    if n == 0:
        return np.nan, np.nan, np.nan

    ll = np.full(n, np.nan, dtype=float)
    rr = np.full(n, np.nan, dtype=float)
    ll[:len(l)] = l
    rr[:len(r)] = r

    if mode == "mean":
        bg = np.nanmean(np.vstack([ll, rr]), axis=0)
    elif mode == "median":
        bg = np.nanmedian(np.vstack([ll, rr]), axis=0)
    else:
        raise ValueError(f"Unknown left/right mode: {mode}")

    bg = bg[np.isfinite(bg)]
    if len(bg) == 0:
        return np.nan, np.nan, np.nan

    left_total = np.nansum(ll)
    right_total = np.nansum(rr)
    asym = abs(left_total - right_total) / (left_total + right_total + 1e-12)

    return float(np.nansum(bg)), float(asym), float(np.nanmedian(bg))


def cosine_similarity_vectors(observed, expected, min_pairs: int = 2) -> float:
    """
    Cosine similarity between observed and expected ion-intensity vectors.

    Both vectors are aligned by position. Non-finite pairs are removed.
    The expected vector is NOT re-scaled using the observed signal; cosine is
    scale-invariant by construction.

    Returns NaN when fewer than min_pairs valid channels remain or either
    vector has zero norm.
    """
    x = _safe_parse_vector(observed)
    w = _safe_parse_vector(expected)

    if len(x) == 0 or len(w) == 0:
        return np.nan

    n = min(len(x), len(w))
    x = x[:n]
    w = w[:n]

    ok = np.isfinite(x) & np.isfinite(w)
    if ok.sum() < min_pairs:
        return np.nan

    x = np.asarray(x[ok], dtype=float)
    w = np.asarray(w[ok], dtype=float)

    nx = np.linalg.norm(x)
    nw = np.linalg.norm(w)

    if nx <= 0 or nw <= 0:
        return np.nan

    return float(np.dot(x, w) / (nx * nw))


def topk_cosine_similarity(
    observed,
    expected,
    k: int = 5,
    min_pairs: int = 2,
) -> float:
    """
    Cosine similarity using the top-k expected channels.

    Useful for MS2 because a small number of strong library fragments can
    dominate precursor identity, while very weak tail fragments contribute
    little reliable information.
    """
    x = _safe_parse_vector(observed)
    w = _safe_parse_vector(expected)

    if len(x) == 0 or len(w) == 0:
        return np.nan

    n = min(len(x), len(w))
    x = x[:n]
    w = w[:n]

    ok = np.isfinite(x) & np.isfinite(w)
    if ok.sum() < min_pairs:
        return np.nan

    x = x[ok]
    w = w[ok]

    kk = min(k, len(w))
    if kk < min_pairs:
        return np.nan

    order = np.argsort(w)[::-1][:kk]
    xx = x[order]
    ww = w[order]

    nx = np.linalg.norm(xx)
    nw = np.linalg.norm(ww)
    if nx <= 0 or nw <= 0:
        return np.nan

    return float(np.dot(xx, ww) / (nx * nw))


def add_ion_pattern_cosines(
    df: pd.DataFrame,
    ms1_signal_col: str = "MS1.XIC.Quant",
    ms1_expected_col: str = "MS1.Rel.Intensity",
    ms2_signal_col: str = "MS2.XIC.Quant",
    ms2_expected_col: str = "MS2.Rel.Intensity",
    ms2_top_k: int = 5,
) -> pd.DataFrame:
    """
    Add whole-vector isotope/fragment pattern cosine similarity.

    Required source columns:
      MS1.XIC.Quant + MS1.Rel.Intensity
      MS2.XIC.Quant + MS2.Rel.Intensity

    The calculation is independent of XIC shape correlation.
    """
    df = df.copy()

    _require(
        df,
        [ms1_signal_col, ms1_expected_col, ms2_signal_col, ms2_expected_col],
        "df",
    )

    df["ms1_isotope_cosine"] = [
        cosine_similarity_vectors(o, e)
        for o, e in zip(df[ms1_signal_col], df[ms1_expected_col])
    ]

    df["ms2_fragment_cosine"] = [
        cosine_similarity_vectors(o, e)
        for o, e in zip(df[ms2_signal_col], df[ms2_expected_col])
    ]

    df["ms2_fragment_cosine_top5"] = [
        topk_cosine_similarity(o, e, k=ms2_top_k)
        for o, e in zip(df[ms2_signal_col], df[ms2_expected_col])
    ]

    return df

def add_precursor_run_features(
    xic_df: pd.DataFrame,
    reported_col="Reported",
    precursor_id_col="Precursor.Id",
    run_id_col="Run",
    tier_col="Evidence.Tier",
    fwhm_col="FWHM",
    rt_query_col="RT.Query",
    rt_apex_col="RT.Apex",
    apex_distance_col="Apex.Distance",
    ms1_candidate_apex_col="MS1.Candidate.Apex",
    ms1_fwhm_left_col="MS1.FWHM.Left",
    ms1_fwhm_right_col="MS1.FWHM.Right",
    ms1_n_usable_region_col="MS1.N.Usable.XICs.Region",
    ms2_n_usable_region_col="MS2.N.Usable.XICs.Region",
    ms1_n_usable_local_col="MS1.N.Usable.XICs.Local",
    ms2_n_usable_local_col="MS2.N.Usable.XICs.Local",
    ms1_signal_col="MS1.XIC.Quant",
    ms1_left_col="MS1.XIC.Quant.Left",
    ms1_right_col="MS1.XIC.Quant.Right",
    ms1_corr_col="MS1.XIC.Corr",
    ms2_signal_col="MS2.XIC.Quant",
    ms2_left_col="MS2.XIC.Quant.Left",
    ms2_right_col="MS2.XIC.Quant.Right",
    ms2_corr_col="MS2.XIC.Corr",
):
    """
    Derive precursor-run-level features from the universal MS1/MS2 XIC extraction.

    No new raw-data extraction is required. Background is computed from the
    supplied adjacent RT flank integrals.

    Output includes:
      - raw linear XIC signals
      - within-run percentile ranks
      - left/right/background integrals
      - background asymmetry
      - XIC correlation/co-elution summaries
      - region-wide and local usable-XIC fractions
      - RT refinement displacement
      - MS1 candidate/FWHM localization diagnostics
      - isotope/fragment pattern cosine similarity

    IMPORTANT:
      Reported=1 is interpreted as original DIA-NN identification/quantification.
    """
    df = xic_df.copy()

    required = [
        reported_col,
        precursor_id_col,
        run_id_col,
        tier_col,
        fwhm_col,
        rt_query_col,
        rt_apex_col,
        apex_distance_col,
        ms1_candidate_apex_col,
        ms1_fwhm_left_col,
        ms1_fwhm_right_col,
        ms1_n_usable_region_col,
        ms2_n_usable_region_col,
        ms1_n_usable_local_col,
        ms2_n_usable_local_col,
        ms1_signal_col,
        ms1_left_col,
        ms1_right_col,
        ms1_corr_col,
        ms2_signal_col,
        ms2_left_col,
        ms2_right_col,
        ms2_corr_col,
    ]
    _require(df, required, "xic_df")

    df["is_anchor"] = df[reported_col].astype(bool)

    # Recovery/localization features derivable from the simple precursor-run table.
    df["evidence_tier_label"] = df[tier_col].map(EVIDENCE_TIER_LABELS).fillna(df[tier_col])
    df["rt_refinement_delta"] = pd.to_numeric(df[rt_apex_col], errors="coerce") - pd.to_numeric(df[rt_query_col], errors="coerce")
    df["rt_refinement_abs_delta"] = df["rt_refinement_delta"].abs()

    cand = pd.to_numeric(df[ms1_candidate_apex_col], errors="coerce")
    left = pd.to_numeric(df[ms1_fwhm_left_col], errors="coerce")
    right = pd.to_numeric(df[ms1_fwhm_right_col], errors="coerce")
    df["ms1_candidate_found"] = cand.notna()
    df["ms1_candidate_fwhm_points"] = right - left + 1
    # Candidate apex is in XIC index space. For prediction-centered rows, Apex.Distance=0
    # and the predicted center is the final apex index; the exact overlap itself is
    # already encoded by the tier and exported FWHM boundaries.
    df["ms1_peak_overlaps_predicted_rt"] = pd.array(
        np.where(
            df[tier_col].eq("ms1_peak"), True,
            np.where(df[tier_col].eq("prediction_only") & cand.notna(), False, pd.NA)
        ), dtype="boolean"
    )

    # Target XIC signals. Use sums in linear intensity space.
    df["ms1_xics_signal"] = df[ms1_signal_col].map(
        lambda x: _vector_summary(x, mode="sum")
    )
    df["ms2_xics_signal"] = df[ms2_signal_col].map(
        lambda x: _vector_summary(x, mode="sum")
    )
    
    # Within-run ranks. These are more appropriate for cross-run phenotyping. Scaled to 0-100
    # (not pandas' native 0-1 pct=True output) so downstream figures read as a percentile.
    for signal_col, rank_col in [
        ("ms1_xics_signal", "ms1_xics_signal_rank"),
        ("ms2_xics_signal", "ms2_xics_signal_rank"),
    ]:
        df[rank_col] = (
            df.groupby(run_id_col, observed=True)[signal_col]
            .rank(pct=True, method="average") * 100
        )
    
    # Ion counts and positive fractions.
    df["ms1_n_isotopes"] = df[ms1_signal_col].map(_vector_count)
    df["ms2_n_fragments"] = df[ms2_signal_col].map(_vector_count)

    # Search-derived usable-XIC counts are exported by JUMPlion using the
    # same peak-search parameters. Derive only parameter-independent fractions
    # here; do not reconstruct usability from the simple report dataframe.
    ms1_usable_region = pd.to_numeric(df[ms1_n_usable_region_col], errors="coerce")
    ms2_usable_region = pd.to_numeric(df[ms2_n_usable_region_col], errors="coerce")
    ms1_usable_local = pd.to_numeric(df[ms1_n_usable_local_col], errors="coerce")
    ms2_usable_local = pd.to_numeric(df[ms2_n_usable_local_col], errors="coerce")

    df["ms1_usable_xic_fraction_region"] = np.where(
        df["ms1_n_isotopes"] > 0,
        ms1_usable_region / df["ms1_n_isotopes"],
        np.nan,
    )
    df["ms2_usable_xic_fraction_region"] = np.where(
        df["ms2_n_fragments"] > 0,
        ms2_usable_region / df["ms2_n_fragments"],
        np.nan,
    )
    df["ms1_usable_xic_fraction_local"] = np.where(
        df["ms1_n_isotopes"] > 0,
        ms1_usable_local / df["ms1_n_isotopes"],
        np.nan,
    )
    df["ms2_usable_xic_fraction_local"] = np.where(
        df["ms2_n_fragments"] > 0,
        ms2_usable_local / df["ms2_n_fragments"],
        np.nan,
    )

    # Useful diagnostics: local availability cannot exceed region-wide
    # availability when both use the same usability definition.
    df["ms1_usable_xic_local_region_ratio"] = np.where(
        ms1_usable_region > 0, ms1_usable_local / ms1_usable_region, np.nan
    )
    df["ms2_usable_xic_local_region_ratio"] = np.where(
        ms2_usable_region > 0, ms2_usable_local / ms2_usable_region, np.nan
    )

    df["ms1_positive_fraction"] = df[ms1_signal_col].map(
        _vector_positive_fraction
    )
    df["ms2_positive_fraction"] = df[ms2_signal_col].map(
        _vector_positive_fraction
    )

    # XIC correlation summaries.
    df["ms1_corr_median"] = df[ms1_corr_col].map(_corr_summary)
    df["ms1_corr_mean"] = df[ms1_corr_col].map(
        lambda x: _corr_summary(x, mode="mean", exclude_zero=False)
    )

    df["ms2_corr_median"] = df[ms2_corr_col].map(_corr_summary)
    df["ms2_corr_mean"] = df[ms2_corr_col].map(
        lambda x: _corr_summary(x, mode="mean", exclude_zero=False)
    )

    # Sum variant (zero-inclusive, same convention as the mean above): total
    # per-channel co-elution rather than its average, so precursors with more
    # corroborating isotope/fragment channels score higher.
    df["ms1_corr_sum"] = df[ms1_corr_col].map(
        lambda x: _corr_summary(x, mode="sum", exclude_zero=False)
    )
    df["ms2_corr_sum"] = df[ms2_corr_col].map(
        lambda x: _corr_summary(x, mode="sum", exclude_zero=False)
    )

    # Reader-facing mechanism aliases. Zero-inclusive means retain the absence
    # of informative correlation as part of the chromatographic-coherence score.
    df["ms1_coelution_score"] = df["ms1_corr_mean"]
    df["ms2_coelution_score"] = df["ms2_corr_mean"]

    # Left/right adjacent RT background.
    ms1_bg = df.apply(
        lambda row: _paired_left_right_summary(
            row[ms1_left_col], row[ms1_right_col], mode="mean"
        ),
        axis=1,
        result_type="expand",
    )
    ms1_bg.columns = [
        "ms1_background",
        "ms1_background_asymmetry",
        "ms1_background_median_ion",
    ]
    df = pd.concat([df, ms1_bg], axis=1)

    ms2_bg = df.apply(
        lambda row: _paired_left_right_summary(
            row[ms2_left_col], row[ms2_right_col], mode="mean"
        ),
        axis=1,
        result_type="expand",
    )
    ms2_bg.columns = [
        "ms2_background",
        "ms2_background_asymmetry",
        "ms2_background_median_ion",
    ]
    df = pd.concat([df, ms2_bg], axis=1)

    df["ms1_target_background_ratio"] = (
        df["ms1_xics_signal"] /
        (df["ms1_background"] + 1e-12)
    )
    df["ms2_target_background_ratio"] = (
        df["ms2_xics_signal"] /
        (df["ms2_background"] + 1e-12)
    )
    
    # Add target/background fractions. F = T / (T + B + epsilon)
    epsilon=1e-12
    df["ms1_target_background_fraction"] = (
        pd.to_numeric(df["ms1_xics_signal"], errors="coerce")
        /
        (
            pd.to_numeric(df["ms1_xics_signal"], errors="coerce")
            + pd.to_numeric(df["ms1_background"], errors="coerce")
            + epsilon
        )
    ).clip(0.0, 1.0)

    df["ms2_target_background_fraction"] = (
        pd.to_numeric(df["ms2_xics_signal"], errors="coerce")
        /
        (
            pd.to_numeric(df["ms2_xics_signal"], errors="coerce")
            + pd.to_numeric(df["ms2_background"], errors="coerce")
            + epsilon
        )
    ).clip(0.0, 1.0)
    
    df = add_ion_pattern_cosines(df)
    
    return df
