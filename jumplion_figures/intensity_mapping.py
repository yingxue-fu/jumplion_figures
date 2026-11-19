
"""
DIA-NN/JUMPlion intensity mapping, per row of a quantity matrix (e.g. one precursor or one
protein -- see `id_col`).

Goal
----
Keep the original DIA-NN quantity matrix whenever possible.

For each row with missing DIA-NN values, fit one linear model using
overlapping observed DIA-NN and JUMPlion quantities:

    log2(DIA-NN) = intercept + slope * log2(JUMPlion)

The model is accepted only when:
    - enough overlapping observed points exist;
    - slope is positive and reasonable;
    - R2 is high enough;
    - MAE is low enough;
    - Pearson/Spearman are sufficiently strong.

Only missing DIA-NN values are predicted.
Original DIA-NN observed values are NEVER overwritten.

Main outputs
------------
Each written file's stem is prefixed by `id_col` (see `_id_col_filename_prefix`) -- "precursor"
for the default "Precursor.Id" (unchanged from before `id_col` existed), or a generic slug of
`id_col` otherwise (e.g. "Protein.Group" -> "protein_group"), so two calls sharing one
`output_dir` with different `id_col`s never overwrite each other's files.

<prefix>_mapping_model_metrics.tsv
    One row per ID with model quality metrics.

<prefix>_quantity_diann_jumplion_recoverd.tsv
    DIA-NN matrix after filling only eligible missing cells.

<prefix>_recover_summary.tsv
    Per-ID recover status and number of values recoverd.

Optional:
    recover_plot_data.tsv
    Compact observed/predicted diagnostic data for selected IDs.
"""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd


def _id_col_filename_prefix(id_col):
    """Output-filename prefix for one `id_col` -- "Precursor.Id" (this module's own default, and
    every output filename's original hardcoded prefix) keeps the exact original "precursor" stem
    for backward compatibility; any other `id_col` (e.g. "Protein.Group" for a protein-level
    quantity matrix) gets a generic slug instead ("protein_group") -- so fit_linear_models'/
    recover_missing_diann's own written files never collide between two `id_col`s sharing the same
    `output_dir`."""
    if id_col == "Precursor.Id":
        return "precursor"
    return re.sub(r"[^0-9a-zA-Z]+", "_", id_col).strip("_").lower()


def load_table(obj):
    if isinstance(obj, pd.DataFrame):
        return obj.copy()

    path = Path(obj)
    if not path.exists():
        raise FileNotFoundError(path)

    suffix = path.suffix.lower()

    if suffix == ".parquet":
        return pd.read_parquet(path)

    if suffix == ".csv":
        return pd.read_csv(path)

    return pd.read_csv(path, sep="\t")


def load_quantity_matrix(
    obj,
    id_col="Precursor.Id",
):
    """
    Load a wide precursor x run quantity matrix.

    Supports either:
        Precursor.Id | Run1 | Run2 | ...

    or an already indexed DataFrame:
        index.name == Precursor.Id
    """
    if isinstance(obj, pd.DataFrame):
        q = obj.copy()

        if (
            q.index.name == id_col
            and id_col not in q.columns
        ):
            q.columns = q.columns.astype(str)
            return q.apply(
                pd.to_numeric,
                errors="coerce",
            )

    q = load_table(obj)
    prot_cols = ['Protein.Group', 'Protein.Ids', 'Protein.Names', 'Genes']

    if id_col not in q.columns:
        aliases = [
            "Precursor.Id",
            "Precursor.ID",
            "PrecursorID",
            "Precursor",
            "index",
            "Index",
            "__index_level_0__",
        ]

        found_col = next(
            (
                c for c in aliases
                if c in q.columns
            ),
            None,
        )

        if found_col is None:
            unnamed = [
                c for c in q.columns
                if str(c).startswith("Unnamed:")
            ]

            if len(unnamed) == 1:
                candidate = unnamed[0]

                s = q[candidate]

                if (
                    s.nunique(dropna=True)
                    / max(len(s), 1)
                    > 0.90
                ):
                    found_col = candidate

        if found_col is None:
            raise KeyError(
                f"Quantity matrix is missing "
                f"{id_col!r}. "
                f"Available columns: {list(q.columns)}"
            )

        q = q.rename(
            columns={
                found_col: id_col
            }
        )

    q = q.drop_duplicates(
        subset=[id_col],
        keep="first",
    )
    
    exclude = {id_col, *prot_cols}
    run_columns = [c for c in q.columns if c not in exclude]

    if not run_columns:
        raise ValueError(
            "No run columns found in quantity matrix."
        )
    
    q = q.set_index(
        id_col
    )
    q = q[run_columns]
    q.columns = q.columns.astype(str)

    return q.apply(
        pd.to_numeric,
        errors="coerce",
    )


def fast_linear_fit(
    x_jump,
    y_diann,
):
    """
    Fit y = intercept + slope*x by direct NumPy formulas.

    Inputs are already log2 transformed and paired.

    Returns
    -------
    dict
        Model coefficients and fit diagnostics.
    """
    x = np.asarray(
        x_jump,
        dtype=float,
    ).ravel()

    y = np.asarray(
        y_diann,
        dtype=float,
    ).ravel()

    valid = (
        np.isfinite(x)
        & np.isfinite(y)
    )

    x = x[valid]
    y = y[valid]

    n = len(x)

    result = {
        "N.Overlap": n,
        "Slope": np.nan,
        "Intercept": np.nan,
        "R2": np.nan,
        "RMSE": np.nan,
        "MAE": np.nan,
        "Pearson": np.nan,
        "Spearman": np.nan,
        "Model.Valid": False,
    }

    if n < 3:
        return result, None

    x_mean = float(
        np.mean(x)
    )
    y_mean = float(
        np.mean(y)
    )

    dx = x - x_mean
    dy = y - y_mean

    ss_x = float(
        np.dot(dx, dx)
    )

    if not np.isfinite(ss_x) or ss_x <= 0:
        return result, None

    slope = float(
        np.dot(dx, dy) / ss_x
    )

    intercept = float(
        y_mean - slope * x_mean
    )

    prediction = (
        intercept
        + slope * x
    )

    residual = (
        y - prediction
    )

    ss_res = float(
        np.dot(
            residual,
            residual,
        )
    )

    ss_tot = float(
        np.dot(
            dy,
            dy,
        )
    )

    if ss_tot > 0:
        r2 = float(
            1.0 - ss_res / ss_tot
        )
    else:
        r2 = np.nan

    rmse = float(
        np.sqrt(
            np.mean(
                residual ** 2
            )
        )
    )

    mae = float(
        np.mean(
            np.abs(residual)
        )
    )

    pearson = np.nan

    if n >= 3:
        sx = np.std(
            x,
            ddof=1,
        )
        sy = np.std(
            y,
            ddof=1,
        )

        if (
            np.isfinite(sx)
            and np.isfinite(sy)
            and sx > 0
            and sy > 0
        ):
            pearson = float(
                np.corrcoef(
                    x,
                    y,
                )[0, 1]
            )

    # Fast Spearman rank correlation.
    spearman = np.nan

    if n >= 3:
        x_rank = (
            pd.Series(x)
            .rank(
                method="average"
            )
            .to_numpy(
                dtype=float
            )
        )

        y_rank = (
            pd.Series(y)
            .rank(
                method="average"
            )
            .to_numpy(
                dtype=float
            )
        )

        sx = np.std(
            x_rank,
            ddof=1,
        )

        sy = np.std(
            y_rank,
            ddof=1,
        )

        if (
            np.isfinite(sx)
            and np.isfinite(sy)
            and sx > 0
            and sy > 0
        ):
            spearman = float(
                np.corrcoef(
                    x_rank,
                    y_rank,
                )[0, 1]
            )

    result.update({
        "Slope": slope,
        "Intercept": intercept,
        "R2": r2,
        "RMSE": rmse,
        "MAE": mae,
        "Pearson": pearson,
        "Spearman": spearman,
        "Model.Valid": (
            np.isfinite(slope)
            and slope > 0
        ),
    })

    return result, {
        "x": x,
        "y": y,
        "prediction": prediction,
        "residual": residual,
    }


def evaluate_model_quality(
    metrics,
    *,
    effective_min_overlap=5,
    min_r2=0.70,
    max_mae=0.50,
    min_spearman=0.80,
    min_slope=0.50,
    max_slope=2.00,
):
    """
    Apply the production model-quality gate and return a primary reason.

    The reason is hierarchical: the first failed criterion is reported.
    This keeps Model.Fail.Reason mutually exclusive for downstream summaries.

    Returns
    -------
    tuple[bool, str]
        (Model.Good, Model.Fail.Reason). Successful models return ``PASS``.
    """
    if metrics["N.Overlap"] < effective_min_overlap:
        return False, "INSUFFICIENT_OVERLAP"

    slope = metrics["Slope"]
    r2 = metrics["R2"]
    mae = metrics["MAE"]
    spearman = metrics["Spearman"]

    if not np.isfinite(slope):
        return False, "INVALID_SLOPE"

    if slope < min_slope:
        return False, "SLOPE_TOO_LOW"

    if slope > max_slope:
        return False, "SLOPE_TOO_HIGH"

    if not np.isfinite(r2):
        return False, "INVALID_R2"

    if r2 < min_r2:
        return False, "LOW_R2"

    if not np.isfinite(mae):
        return False, "INVALID_MAE"

    if mae > max_mae:
        return False, "HIGH_MAE"

    # Preserve the original behavior: a non-finite Spearman does not fail
    # the model; only a finite value below the threshold does.
    if np.isfinite(spearman) and spearman < min_spearman:
        return False, "LOW_SPEARMAN"

    return True, "PASS"


def is_model_good(
    metrics,
    *,
    effective_min_overlap=5,
    min_r2=0.70,
    max_mae=0.50,
    min_spearman=0.80,
    min_slope=0.50,
    max_slope=2.00,
):
    """Backward-compatible boolean wrapper around evaluate_model_quality."""
    good, _ = evaluate_model_quality(
        metrics,
        effective_min_overlap=effective_min_overlap,
        min_r2=min_r2,
        max_mae=max_mae,
        min_spearman=min_spearman,
        min_slope=min_slope,
        max_slope=max_slope,
    )
    return good


def fit_linear_models(
    quantity_diann,
    quantity_jumplion,
    *,
    id_col="Precursor.Id",
    min_overlap=5,
    min_overlap_fraction=0.3,
    min_r2=0.70,
    max_mae=0.50,
    min_spearman=0.80,
    min_slope=0.50,
    max_slope=2.00,
    output_dir=None,
):
    """
    Fit one fast direct linear mapping model per precursor.

    Only precursors with at least one missing DIA-NN value need a model,
    which saves substantial runtime.
    """
    diann = load_quantity_matrix(
        quantity_diann,
        id_col=id_col,
    )

    jumplion = load_quantity_matrix(
        quantity_jumplion,
        id_col=id_col,
    )

    common_precursors = diann.index.intersection(
        jumplion.index
    )

    common_runs = diann.columns.intersection(
        jumplion.columns
    )

    if len(common_precursors) == 0:
        raise ValueError(
            "No common precursors."
        )

    if len(common_runs) == 0:
        raise ValueError(
            "No common runs."
        )

    n_runs = len(common_runs)
    effective_min_overlap = max(
        min_overlap,
        math.ceil(min_overlap_fraction * n_runs),
    )

    rows = []
    models = {}

    n_skipped_no_missing = 0

    # Convert once to NumPy for faster row processing.
    diann_common = diann.loc[
        common_precursors,
        common_runs,
    ].to_numpy(dtype=float)

    jump_common = jumplion.loc[
        common_precursors,
        common_runs,
    ].to_numpy(dtype=float)

    for i, precursor in enumerate(
        common_precursors
    ):
        y = diann_common[i]
        x = jump_common[i]

        # No DIA-NN missing values: no mapping needed.
        missing_y = (
            ~np.isfinite(y)
        )

        if not missing_y.any():
            n_skipped_no_missing += 1
            continue

        valid = (
            np.isfinite(x)
            & np.isfinite(y)
            & (x > 0)
            & (y > 0)
        )

        n_overlap = int(
            valid.sum()
        )
        
        n_overlap_fraction = n_overlap / len(common_runs)
        
        if n_overlap < effective_min_overlap:
            rows.append({
                id_col: precursor,
                "N.Overlap": n_overlap,
                "N.Overlap.Fraction": n_overlap_fraction,
                "Effective.Min.Overlap": effective_min_overlap,
                "Slope": np.nan,
                "Intercept": np.nan,
                "R2": np.nan,
                "RMSE": np.nan,
                "MAE": np.nan,
                "Pearson": np.nan,
                "Spearman": np.nan,
                "Model.Valid": False,
                "Model.Good": False,
                "Model.Fail.Reason": "INSUFFICIENT_OVERLAP",
                "N.DIANN.Missing": int(
                    missing_y.sum()
                ),
                "Recover.Possible": False,
            })
            continue

        x_log = np.log2(
            x[valid]
        )

        y_log = np.log2(
            y[valid]
        )

        metrics, model_data = fast_linear_fit(
            x_log,
            y_log,
        )
        
        metrics["N.Overlap.Fraction"] = n_overlap_fraction
        metrics["Effective.Min.Overlap"] = effective_min_overlap
        
        good, fail_reason = evaluate_model_quality(
            metrics,
            effective_min_overlap=effective_min_overlap,
            min_r2=min_r2,
            max_mae=max_mae,
            min_spearman=min_spearman,
            min_slope=min_slope,
            max_slope=max_slope,
        )

        metrics["Model.Good"] = bool(good)
        metrics["Model.Fail.Reason"] = fail_reason
        metrics["N.DIANN.Missing"] = int(
            missing_y.sum()
        )
        metrics["Recover.Possible"] = bool(
            good
            and np.any(
                missing_y
                & np.isfinite(x)
                & (x > 0)
            )
        )

        rows.append({
            id_col: precursor,
            **metrics,
        })

        if good:
            models[precursor] = {
                "slope": metrics["Slope"],
                "intercept": metrics["Intercept"],
            }

    metrics_df = pd.DataFrame(
        rows
    )

    if output_dir is not None:
        outdir = Path(output_dir)
        outdir.mkdir(
            parents=True,
            exist_ok=True,
        )
        prefix = _id_col_filename_prefix(id_col)

        metrics_df.to_csv(
            outdir
            / f"{prefix}_mapping_model_metrics.tsv",
            sep="\t",
            index=False,
        )

        pd.DataFrame({
            "metric": [
                "n_common_precursors",
                "n_precursors_with_DIA_NN_missing",
                "n_good_models",
                "fraction_good_among_missing_precursors",
                "effective_min_overlap",
                "median_N_overlap",
                "median_R2",
                "median_MAE",
                "median_Spearman",
                "median_slope",
            ],
            "value": [
                len(common_precursors),
                int(
                    (metrics_df["N.DIANN.Missing"] > 0)
                    .sum()
                ),
                int(
                    metrics_df["Model.Good"]
                    .fillna(False)
                    .sum()
                ),
                (
                    metrics_df["Model.Good"]
                    .fillna(False)
                    .mean()
                    if len(metrics_df)
                    else np.nan
                ),
                effective_min_overlap,
                metrics_df["N.Overlap"].median(),
                metrics_df["R2"].median(),
                metrics_df["MAE"].median(),
                metrics_df["Spearman"].median(),
                metrics_df["Slope"].median(),
            ],
        }).to_csv(
            outdir / f"{prefix}_mapping_summary.tsv",
            sep="\t",
            index=False,
        )

    return {
        "models": models,
        "metrics": metrics_df,
        "diann": diann,
        "jumplion": jumplion,
        "common_precursors": common_precursors,
        "common_runs": common_runs,
    }


def recover_missing_diann(
    model_result,
    *,
    id_col="Precursor.Id",
    min_missing_values_for_recover=2,
    min_missing_fraction_for_recover=0.0,
    max_extrapolation_log2: Optional[float] = 0.5,
    clip_prediction_log2: Optional[
        Tuple[float, float]
    ] = None,
    output_dir=None,
):
    """
    Predict missing DIA-NN values using good precursor-specific models.

    Original DIA-NN values are never modified.

    A precursor with fewer than
    `min_missing_values_for_recover` missing DIA-NN values is left unchanged.

    Prediction applicability is constrained by `max_extrapolation_log2`. A missing
    precursor-run is left unresolved when its log2 JUMPlion raw quantity lies farther
    than this distance outside the observed JUMPlion range used to fit the model.
    Set to None to disable the extrapolation safeguard.
    """
    diann = model_result["diann"].copy()
    jumplion = model_result["jumplion"]
    models = model_result["models"]

    out = diann.copy()
    rows = []

    common_runs = diann.columns.intersection(
        jumplion.columns
    )

    for precursor in diann.index:
        n_missing = int(
            diann.loc[
                precursor
            ].isna().sum()
        )

        summary = {
            id_col: precursor,
            "N.DIANN.Missing": n_missing,
            "Missing.Fraction": (
                n_missing
                / max(
                    len(diann.columns),
                    1,
                )
            ),
            "Model.Good": precursor in models,
            "N.Recoverd": 0,
            "Recover.Applied": False,
            "Recover.Reason": "",
            "N.Predictable.Before.Extrapolation.Filter": 0,
            "N.Excluded.By.Extrapolation": 0,
            "Max.Extrapolation.Distance.log2": np.nan,
        }

        if precursor not in models:
            summary["Recover.Reason"] = (
                "MODEL_NOT_GOOD"
            )
            rows.append(summary)
            continue

        if n_missing < min_missing_values_for_recover:
            summary["Recover.Reason"] = (
                "TOO_FEW_MISSING_VALUES"
            )
            rows.append(summary)
            continue

        missing_fraction = (
            n_missing
            / max(
                len(diann.columns),
                1,
            )
        )

        if (
            missing_fraction
            < min_missing_fraction_for_recover
        ):
            summary["Recover.Reason"] = (
                "MISSING_FRACTION_TOO_LOW"
            )
            rows.append(summary)
            continue

        jump_row = jumplion.loc[
            precursor,
            common_runs,
        ]

        diann_row = diann.loc[
            precursor,
            common_runs,
        ]

        jump_arr = jump_row.to_numpy(
            dtype=float
        )

        diann_arr = diann_row.to_numpy(
            dtype=float
        )

        mask = (
            ~np.isfinite(diann_arr)
            & np.isfinite(jump_arr)
            & (jump_arr > 0)
        )

        if not mask.any():
            summary["Recover.Reason"] = (
                "NO_PREDICTABLE_MISSING_CELLS"
            )
            rows.append(summary)
            continue

        model = models[
            precursor
        ]

        # Prediction applicability is evaluated separately from Model.Good.
        # The mapping model is fitted on paired observed runs; individual missing
        # precursor-runs are recovered only when their raw JUMPlion quantity is
        # within max_extrapolation_log2 of the observed fitting range in log2 space.
        fit_mask = (
            np.isfinite(diann_arr)
            & (diann_arr > 0)
            & np.isfinite(jump_arr)
            & (jump_arr > 0)
        )
        fit_x_log2 = np.log2(jump_arr[fit_mask])
        fit_x_min = float(np.min(fit_x_log2))
        fit_x_max = float(np.max(fit_x_log2))

        candidate_indices = np.flatnonzero(mask)
        candidate_x_log2 = np.log2(jump_arr[candidate_indices])
        extrapolation_distance = np.maximum(
            fit_x_min - candidate_x_log2,
            np.maximum(candidate_x_log2 - fit_x_max, 0.0),
        )

        summary["N.Predictable.Before.Extrapolation.Filter"] = int(len(candidate_indices))
        summary["Max.Extrapolation.Distance.log2"] = (
            float(np.max(extrapolation_distance))
            if len(extrapolation_distance)
            else np.nan
        )

        if max_extrapolation_log2 is None:
            supported = np.ones(len(candidate_indices), dtype=bool)
        else:
            supported = extrapolation_distance <= float(max_extrapolation_log2)

        summary["N.Excluded.By.Extrapolation"] = int((~supported).sum())
        candidate_indices = candidate_indices[supported]
        candidate_x_log2 = candidate_x_log2[supported]

        if len(candidate_indices) == 0:
            summary["Recover.Reason"] = "ALL_PREDICTIONS_EXCEED_EXTRAPOLATION_LIMIT"
            rows.append(summary)
            continue

        pred_log2 = (
            model["intercept"]
            + model["slope"] * candidate_x_log2
        )

        if clip_prediction_log2 is not None:
            pred_log2 = np.clip(
                pred_log2,
                clip_prediction_log2[0],
                clip_prediction_log2[1],
            )

        predicted = np.power(
            2.0,
            pred_log2,
        )

        runs = common_runs.to_numpy()

        recover_n = 0

        for run, value in zip(
            runs[candidate_indices],
            predicted,
        ):
            if pd.isna(
                out.loc[
                    precursor,
                    run,
                ]
            ):
                out.loc[
                    precursor,
                    run,
                ] = float(value)

                recover_n += 1

        summary["N.Recoverd"] = recover_n
        summary["Recover.Applied"] = (
            recover_n > 0
        )
        if recover_n > 0 and summary["N.Excluded.By.Extrapolation"] > 0:
            summary["Recover.Reason"] = "GOOD_MODEL_PARTIAL_EXTRAPOLATION_FILTER"
        elif recover_n > 0:
            summary["Recover.Reason"] = "GOOD_MODEL"
        else:
            summary["Recover.Reason"] = "NO_RESCUE_APPLIED"

        rows.append(summary)

    recover_summary = pd.DataFrame(
        rows
    )

    if output_dir is not None:
        outdir = Path(output_dir)
        outdir.mkdir(
            parents=True,
            exist_ok=True,
        )
        prefix = _id_col_filename_prefix(id_col)

        out.reset_index().to_csv(
            outdir
            / f"{prefix}_quantity_diann_jumplion_recoverd.tsv",
            sep="\t",
            index=False,
        )

        recover_summary.to_csv(
            outdir
            / f"{prefix}_recover_summary.tsv",
            sep="\t",
            index=False,
        )

    return {
        "quantity_recoverd": out.reset_index(),
        "recover_summary": recover_summary,
    }


def run_intensity_mapping(
    quantity_diann,
    quantity_jumplion,
    *,
    id_col="Precursor.Id",
    min_overlap=5,
    min_overlap_fraction=0.3,
    min_r2=0.70,
    max_mae=0.50,
    min_spearman=0.80,
    min_slope=0.50,
    max_slope=2.00,
    min_missing_values_for_recover=2,
    min_missing_fraction_for_recover=0.0,
    max_extrapolation_log2: Optional[float] = 0.5,
    clip_prediction_log2: Optional[
        Tuple[float, float]
    ] = None,
    output_dir=None,
):
    """
    Fast end-to-end mapping and missing-value recover.
    """
    model_result = fit_linear_models(
        quantity_diann,
        quantity_jumplion,
        id_col=id_col,
        min_overlap=min_overlap,
        min_overlap_fraction=min_overlap_fraction,
        min_r2=min_r2,
        max_mae=max_mae,
        min_spearman=min_spearman,
        min_slope=min_slope,
        max_slope=max_slope,
        output_dir=output_dir,
    )

    recover_result = recover_missing_diann(
        model_result,
        id_col=id_col,
        min_missing_values_for_recover=min_missing_values_for_recover,
        min_missing_fraction_for_recover=min_missing_fraction_for_recover,
        max_extrapolation_log2=max_extrapolation_log2,
        clip_prediction_log2=clip_prediction_log2,
        output_dir=output_dir,
    )

    return {
        **model_result,
        **recover_result,
    }


def summarize_models(
    metrics_df,
):
    """Compact model-quality summary."""
    if metrics_df.empty:
        return pd.DataFrame({
            "metric": [
                "n_precursors",
                "n_good_models",
                "fraction_good_models",
            ],
            "value": [
                0,
                0,
                np.nan,
            ],
        })

    good = metrics_df[
        "Model.Good"
    ].fillna(False)

    return pd.DataFrame({
        "metric": [
            "n_precursors_with_missing",
            "n_good_models",
            "fraction_good_models",
            "median_N_overlap",
            "median_R2",
            "median_MAE",
            "median_Spearman",
            "median_slope",
        ],
        "value": [
            len(metrics_df),
            int(good.sum()),
            float(good.mean()),
            metrics_df[
                "N.Overlap"
            ].median(),
            metrics_df[
                "R2"
            ].median(),
            metrics_df[
                "MAE"
            ].median(),
            metrics_df[
                "Spearman"
            ].median(),
            metrics_df.loc[
                good,
                "Slope",
            ].median()
            if good.any()
            else np.nan,
        ],
    })


def define_status(df):
    # 1. Define your logical conditions (order matters: most specific to least specific)
    conditions = [
        # Rescued: All three columns must be present
        (df['DIANN.Quantity'].isna()) & 
        (df['JUMPlion.Raw.Quantity'].notna()) & 
        (df['Final.Quantity'].notna()),
        
        # Observed: DIANN and Final are present (but not JUMPlion, due to the order)
        (df['DIANN.Quantity'].notna()) & 
        (df['Final.Quantity'].notna())
    ]
    
    # 2. Define the choices corresponding to each condition
    choices = ['recovered', 'observed']
    
    # 3. Apply conditions, and set the fallback default to 'unresolved'
    df['Final.Status'] = np.select(conditions, choices, default='unresolved')
    
    return df


def plot_intensity_mapping_workflow():
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch
    
    # 4A: conceptual precursor-specific mapping workflow.
    # Detailed numerical thresholds are intentionally left to the caption/Methods.
    fig, ax = plt.subplots(figsize=(4.8, 6.8))
    ax.axis("off")
    
    
    def workflow_box(x, y, w, h, text, weight="normal", ls="-"):
        patch = FancyBboxPatch(
            (x - w / 2, y - h / 2),
            w,
            h,
            boxstyle="round,pad=0.012",
            transform=ax.transAxes,
            facecolor="white",
            edgecolor="black",
            linewidth=1.2,
            linestyle=ls,
        )
        ax.add_patch(patch)
    
        if text:
            ax.text(
                x,
                y,
                text,
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=9.5,
                fontweight=weight,
            )
    
    
    def workflow_arrow(x1, y1, x2, y2, text=None):
        ax.annotate(
            "",
            xy=(x2, y2),
            xytext=(x1, y1),
            xycoords=ax.transAxes,
            arrowprops=dict(
                arrowstyle="->",
                lw=1.2,
            ),
        )
    
        if text:
            ax.text(
                (x1 + x2) / 2,
                (y1 + y2) / 2,
                text,
                ha="center",
                va="center",
                fontsize=8,
                transform=ax.transAxes,
            )
    
    
    # ------------------------------------------------------------
    # Input and mapping
    # ------------------------------------------------------------
    
    workflow_box(
        0.50, 0.93, 0.66, 0.075,
        "JUMPlion raw precursor quantity",
        weight="bold",
    )
    
    workflow_arrow(0.50, 0.89, 0.50, 0.84)
    
    workflow_box(
        0.50, 0.79, 0.76, 0.095,
        "Paired DIA-NN and JUMPlion\nprecursor quantities",
    )
    
    workflow_arrow(0.50, 0.74, 0.50, 0.69)
    
    workflow_box(
        0.50, 0.64, 0.72, 0.095,
        "Precursor-specific linear mapping\n(log$_2$ scale)",
        weight="bold",
    )
    
    workflow_arrow(0.50, 0.59, 0.50, 0.545)
    
    
    # ------------------------------------------------------------
    # Mapping-quality criteria
    # ------------------------------------------------------------
    
    # One large box represents the combined mapping-quality decision.
    workflow_box(
        0.50, 0.375, 0.76, 0.31, ""
    )
    
    ax.text(
        0.50, 0.495,
        "Mapping quality criteria",
        ha="center",
        va="center",
        transform=ax.transAxes,
        fontsize=10,
        fontweight="bold",
    )
    
    # Gate 1
    ax.text(
        0.50, 0.435,
        "Sufficient overlap",
        ha="center",
        va="center",
        transform=ax.transAxes,
        fontsize=9.5,
        fontweight="bold",
    )
    
    # Arrow: Gate 1 -> Gate 2
    workflow_arrow(
        0.50, 0.412,
        0.50, 0.375,
    )
    
    # Gate 2
    ax.text(
        0.50, 0.350,
        "Consistent quantitative changes",
        ha="center",
        va="center",
        transform=ax.transAxes,
        fontsize=9.5,
        fontweight="bold",
    )
    
    # Arrow: Gate 2 -> Gate 3
    workflow_arrow(
        0.50, 0.327,
        0.50, 0.290,
    )
    
    # Gate 3
    ax.text(
        0.50, 0.265,
        "Small residual deviations",
        ha="center",
        va="center",
        transform=ax.transAxes,
        fontsize=9.5,
        fontweight="bold",
    )
    
    
    # ------------------------------------------------------------
    # Outcomes
    # ------------------------------------------------------------
    
    workflow_arrow(
        0.40, 0.215,
        0.27, 0.135,
        "Pass",
    )
    
    workflow_arrow(
        0.60, 0.215,
        0.73, 0.135,
        "Fail",
    )
    
    workflow_box(
        0.25, 0.085, 0.40, 0.09,
        "Mapped precursor\nquantity",
        weight="bold",
    )
    
    workflow_box(
        0.75, 0.085, 0.40, 0.09,
        "Unresolved\n(remains missing)",
        ls="--",
    )
    
    ax.set_title(
        "A  Intensity-mapping mechanism",
        loc="left",
        fontweight="bold",
    )
    
    plt.tight_layout()
