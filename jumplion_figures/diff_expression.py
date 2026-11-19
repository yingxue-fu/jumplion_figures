"""Vendored copy of diff_expression_analysis_with_cv_preDE.py (Aspen's script), self-contained.

Differences from the original script (analysis logic unchanged):
  * add_all_protein_names / define_species (previously imported from the jumplion package) are
    defined here; FASTA parsing uses Biopython.
  * plot_volcano: outcome counts use .get(key, 0) and the FDR division is guarded, so a contrast
    with zero calls of some outcome (e.g. no false positives) no longer raises KeyError.
  * the unused `import polars` was removed.
  * load_data no longer takes a `value_start_col` -- which matrix columns are sample (value)
    columns is derived from the sample metadata file's own sample names instead of a fixed
    positional split, so a matrix whose own column layout changes (an annotation column added,
    removed, or reordered) no longer silently misparses or needs a matching config change.

Differential abundance analysis for a DIA-NN style quant matrix.

Designed to be used two ways:

1. As a standalone script (edit the CONFIG block at the bottom under
   `if __name__ == "__main__"` and run `python diff_expression_analysis.py`).

2. As an importable module from another script:

       import diff_expression_analysis as dea

       mat, sample_to_group = dea.load_data("matrix.tsv", "meta.csv")

       group_pairs = [
           ("H50Y30E20", "H50Y10E40"),
           ("H50Y25E25", "H50Y10E40"),
       ]
       results = dea.run_comparisons(mat, sample_to_group, group_pairs)
       # results is a dict: {"H50Y30E20_vs_H50Y10E40": DataFrame, ...}

Pipeline per group pair:
  1. Filter rows by missingness: kept only if BOTH groups have a missing
     percentage <= max_missing_pct.
  2. Compute log2 fold change (mean of group1 log2 values - mean of group2
     log2 values, using only the non-missing values in each group).
  3. Run Welch's t-test (unequal variance) on the log2-transformed values,
     ignoring missing values (nan_policy='omit').
  4. Apply Benjamini-Hochberg FDR correction.

Install dependencies if needed:
    pip install pandas numpy scipy statsmodels seaborn biopython
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from Bio import SeqIO

def extract_from_single_fasta(fasta_file, protein_ids, extract_sequence=True):
    #logging.info(f"Extracting protein information from FASTA file: {fasta_file}")

    prot_info = {}
    with open(fasta_file) as fasta:
        records = list(SeqIO.parse(fasta, 'fasta'))

        for record in records:
            header = record.description
            temp = header.split(" ", 1)
            prot_id = temp[0].split("|")[1]
            if prot_id in protein_ids:
                prot_data = {
                    'Accession.Number': temp[0],
                    'Protein.Description': temp[1] if len(temp) > 1 else ''
                }
                if extract_sequence:
                    prot_data['Sequence'] = str(record.seq)

                prot_info[prot_id] = prot_data
    return prot_info


def extract_prot_info_from_FASTAs(fasta_files, protein_ids, extract_sequence=True, out_file=0):
    collected_info = {}
    for protein_id in protein_ids:
        collected_info[protein_id] = {
            'Accession.Number': '',
            'Protein.Description': '',
            'Sequence': '' if extract_sequence else ''
        }
    if len(fasta_files) == 1:
        prot_info = extract_from_single_fasta(fasta_files[0], protein_ids, extract_sequence)
    else:
        prot_info = {}
        for fasta_file in fasta_files:
            temp_info = extract_from_single_fasta(fasta_file, protein_ids, extract_sequence)
            prot_info.update(temp_info)

    for prot_id, info in prot_info.items():
        collected_info[prot_id] = info

    prot_info_df = pd.DataFrame.from_dict(collected_info, orient='index').reset_index()
    prot_info_df.rename(columns={'index': 'Protein.Id'}, inplace=True)

    if out_file != 0:
        prot_info_df.to_csv(out_file, index=False)
    return prot_info_df


def add_all_protein_names(df, fasta_files, id_col='Protein.Ids', extract_sequence=False, out_file=0):
    """
    Splits semicolon-separated ids in `id_col`, looks up every individual id
    in the given FASTA file(s), and adds an 'Protein.Names.All' column with
    all corresponding protein names joined by ';' (in the same order as
    the ids in id_col).
    """
    # 1. Collect every individual protein id from the (possibly multi-id) column
    split_ids = df[id_col].astype(str).str.split(';')
    all_ids = set(pid for sublist in split_ids for pid in sublist)

    # 2. Extract info for all of them at once
    prot_info_df = extract_prot_info_from_FASTAs(
        fasta_files, all_ids, extract_sequence=extract_sequence
    )

    # 3. Build id -> name lookup.
    #    'Accession.Number' looks like 'sp|P14550|AK1A1_HUMAN' -> take last field
    #    Fall back to '' if not found in the FASTA.
    def name_from_accession(acc):
        if not acc:
            return ''
        parts = acc.split('|')
        return parts[-1] if len(parts) > 1 else acc

    id_to_name = dict(
        zip(prot_info_df['Protein.Id'], prot_info_df['Accession.Number'].apply(name_from_accession))
    )

    # 4. Map back onto each row, preserving id order, joined by ';'
    def lookup_names(id_string):
        ids = str(id_string).split(';')
        names = [id_to_name.get(pid, '') for pid in ids]
        return ';'.join(names)

    df = df.copy()
    df['Protein.Names.All'] = df[id_col].apply(lookup_names)

    if out_file != 0:
        df.to_csv(out_file, index=False)

    return df

def define_species(prot_names):
    if pd.isna(prot_names) or not prot_names.strip():
        return None

    species = set()
    for prot_name in prot_names.split(';'):
        parts = prot_name.strip().split('_')
        if len(parts) > 1:
            species.add(parts[1].strip())

    return ";".join(sorted(species)) if species else None


def diff_benchmark_analysis(mat_files,
                            mat_names,
                            mat_type,
                            mat_sep,
                            log_transform_input,
                            sample_meta_file,
                            fasta_files,
                            pairs,
                            out_dir,
                            p_adj_cutoff=0.05,
                            log2fc_cutoff=0.5,
                            padj_col="p_value",
                            log2fc_col="log2FC",
                            ):
    for i, file in enumerate(mat_files):
        mat, prot_df, sample_to_group = load_data(file, mat_type, sample_meta_file, matrix_sep=mat_sep)
        if fasta_files is not None:
            prot_df = add_all_protein_names(prot_df, fasta_files, id_col='Protein.Ids', extract_sequence=False, out_file=0)
        prot_df = prot_df.drop_duplicates(subset=[mat_type])
        results, pre_de_results = run_comparisons(
            mat, sample_to_group, pairs, max_missing_pct=0.5,
            log_transform_input=log_transform_input,
            return_pre_de=True,
        )
        
        dea_out_dir = Path(out_dir / f"{mat_names[i]}_dea_results")
        dea_out_dir.mkdir(parents=True, exist_ok=True)  # Create directory if missing
        
        for pair in pairs:
            pair_name = f"{pair[0]}_vs_{pair[1]}"
            de_result = results[pair_name].copy()  # DE-eligible rows only
            de_result.reset_index(names=mat_type, inplace=True)

            # Pre-DE table contains ALL precursors before the >=3-observation
            # per-group eligibility filter. It is useful for quantifying how
            # JUMPlion changes DE testability without weakening the DE rule.
            pre_de_result = pre_de_results[pair_name].copy()
            pre_de_result.reset_index(names=mat_type, inplace=True)
            pre_de_table = pre_de_result.merge(prot_df, on=mat_type, how="left")
            if fasta_files is not None:
                pre_de_table['Species'] = pre_de_table['Protein.Names.All'].apply(define_species)
            else:
                pre_de_table['Species'] = pre_de_table['Protein.Names'].apply(define_species)
            pre_de_path = dea_out_dir / f"{mat_names[i]}_{pair_name}_pre_DE.csv"
            pre_de_table.to_csv(pre_de_path, index=False)
            print(f"Saved {pre_de_table.shape[0]} pre-DE rows to {pre_de_path}")
            #print(prot_df)
            #print(de_result)
            prec_quantity_table = de_result.merge(prot_df, on=mat_type, how="left")
            if fasta_files is not None:
                prec_quantity_table['Species'] = prec_quantity_table['Protein.Names.All'].apply(define_species)
            else:
                prec_quantity_table['Species'] = prec_quantity_table['Protein.Names'].apply(define_species)
            
            prec_quantity_table = classify_de_result(prec_quantity_table, p_adj_cutoff, log2fc_cutoff,
                                                     padj_col, log2fc_col, species_col="Species", 
                                                     result_col="DE_result")
            out_path = dea_out_dir / f"{mat_names[i]}_{pair_name}.csv"
            prec_quantity_table.to_csv(out_path)
            print(f"Saved {prec_quantity_table.shape[0]} results to {out_path}")
            
            fig, ax = plt.subplots(1, 1, sharey=True, figsize=(2.4, 2.4))
            plot_volcano(ax, prec_quantity_table, -3.2, 3.2, 0.5, 0.05, pair_name, 2, 8) # 24 for 480, 22 for timeTOF 
            plt.savefig(dea_out_dir / f"{mat_names[i]}_{pair_name}_volcano.svg", bbox_inches="tight")
            
            #fig, ax = plt.subplots(1, 1, sharey=True, figsize=(2.4, 2.4))
            plot_scatter_box(prec_quantity_table, title=f'{mat_names[i]}_{pair_name}', x_col=pair[1],
                             figsize=(3.2, 2.4), width_ratios=[4, 1])
            plt.savefig(dea_out_dir / f"{mat_names[i]}_{pair_name}_fc.svg", bbox_inches="tight")
                      
    

def load_data(matrix_path, matix_type, meta_path, sample_col="Run", group_col="Group", matrix_sep="\t"):
    """Load the quant matrix and sample metadata once. Which columns are sample (value) columns is
    derived from `meta_path`'s own `sample_col` values, not a fixed positional split -- every
    matrix column whose name matches a metadata sample is a value column; every other column
    (however many ID/annotation columns the matrix has, wherever they fall) is row metadata. This
    is what makes load_data robust to a matrix's own column layout changing between files or over
    time (e.g. one extra or missing annotation column) without a matching config change.

    Returns
    -------
    mat : DataFrame (rows = IDs, cols = samples that have metadata)
    prot_info : DataFrame of every matrix column that ISN'T a sample column
    sample_to_group : Series mapping sample name -> group label
    """
    df = pd.read_csv(matrix_path, sep=matrix_sep)
    meta = pd.read_csv(meta_path)
    meta = meta[[sample_col, group_col]].dropna()

    sample_names = set(meta[sample_col])
    value_cols = [c for c in df.columns if c in sample_names]
    if not value_cols:
        raise ValueError(f"{matrix_path}: no column matches any '{sample_col}' value in {meta_path}.")
    mat = df[value_cols]
    prot_info = df.drop(columns=value_cols)
    mat.index = prot_info[matix_type]

    meta = meta[meta[sample_col].isin(mat.columns)]
    missing_in_matrix = sample_names - set(mat.columns)
    if missing_in_matrix:
        print(f"Warning: {len(missing_in_matrix)} metadata samples have no matching column in "
              f"{matrix_path} and will be skipped: {sorted(missing_in_matrix)}")

    return mat, prot_info, meta.set_index(sample_col)[group_col]


def get_group_samples(sample_to_group, group_name):
    samples = sample_to_group[sample_to_group == group_name].index.tolist()
    if len(samples) == 0:
        raise ValueError(f"No samples found for group '{group_name}'")
    return samples


def filter_by_missingness(mat, g1_samples, g2_samples, max_missing_pct):
    g1_missing_pct = mat[g1_samples].isna().mean(axis=1)
    g2_missing_pct = mat[g2_samples].isna().mean(axis=1)

    keep = (g1_missing_pct <= max_missing_pct) & (g2_missing_pct <= max_missing_pct)

    print(f"  Rows before filtering: {mat.shape[0]}")
    print(f"  Rows after filtering (missing <= {max_missing_pct:.0%} in each group): "
          f"{keep.sum()}")

    return mat.loc[keep], g1_missing_pct.loc[keep], g2_missing_pct.loc[keep]


def compute_log2(mat, log_transform=True, pseudocount=None):
    if not log_transform:
        return mat
    if pseudocount is not None:
        return np.log2(mat + pseudocount)
    # log2 of zeros/negatives -> NaN, which is fine, they're treated as missing
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.log2(mat.where(mat > 0))


def run_differential_analysis(log_mat, g1_samples, g2_samples, group1_name, group2_name):
    g1 = log_mat[g1_samples].to_numpy(dtype=float)
    g2 = log_mat[g2_samples].to_numpy(dtype=float)

    g1_mean = np.nanmean(g1, axis=1)
    g2_mean = np.nanmean(g2, axis=1)
    log2fc = g1_mean - g2_mean

    # Group CVs on the original (linear) intensity scale.
    # log_mat contains log2 intensities at this point, so convert back to
    # linear scale before calculating CV = SD / mean. ddof=1 gives the
    # sample SD across biological/technical replicates. Rows with fewer
    # than 2 observed values, or a non-positive mean, are reported as NaN.
    g1_linear = np.exp2(g1)
    g2_linear = np.exp2(g2)
    g1_n = np.sum(np.isfinite(g1_linear), axis=1)
    g2_n = np.sum(np.isfinite(g2_linear), axis=1)
    g1_linear_mean = np.nanmean(g1_linear, axis=1)
    g2_linear_mean = np.nanmean(g2_linear, axis=1)
    g1_sd = np.nanstd(g1_linear, axis=1, ddof=1)
    g2_sd = np.nanstd(g2_linear, axis=1, ddof=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        g1_cv = g1_sd / g1_linear_mean
        g2_cv = g2_sd / g2_linear_mean
    g1_cv[(g1_n < 2) | (g1_linear_mean <= 0)] = np.nan
    g2_cv[(g2_n < 2) | (g2_linear_mean <= 0)] = np.nan

    # Welch's t-test row by row, ignoring NaNs
    t_stat, p_val = stats.ttest_ind(g1, g2, axis=1, equal_var=False, nan_policy="omit")

    results = pd.DataFrame({
        "log2FC": log2fc,
        f"mean_log2_{group1_name}": g1_mean,
        f"mean_log2_{group2_name}": g2_mean,
        f"cv_{group1_name}": g1_cv,
        f"cv_{group2_name}": g2_cv,
        f"n_obs_{group1_name}": g1_n,
        f"n_obs_{group2_name}": g2_n,
        "t_stat": t_stat,
        "p_value": p_val,
    }, index=log_mat.index)

    # BH FDR correction (skip rows where p-value is NaN, e.g. too few points)
    valid = results["p_value"].notna()
    results["p_adj"] = np.nan
    if valid.sum() > 0:
        results.loc[valid, "p_adj"] = multipletests(
            results.loc[valid, "p_value"], method="fdr_bh"
        )[1]

    return results.sort_values("p_value")


def build_pre_de_table(mat, g1_samples, g2_samples, group1_name, group2_name,
                       max_missing_pct=0.5, log_transform_input=True, pseudocount=None):
    """Summarize ALL precursors before the DE eligibility filter.

    The returned table does not contain t-test or adjusted-p values. Instead,
    it records group means, CVs, observed replicate counts, missing fractions,
    and whether each precursor is eligible for DE testing under the same
    missingness rule used by compare_groups().
    """
    log_mat = compute_log2(mat, log_transform_input, pseudocount)
    g1 = log_mat[g1_samples].to_numpy(dtype=float)
    g2 = log_mat[g2_samples].to_numpy(dtype=float)
    g1_n = np.sum(np.isfinite(g1), axis=1)
    g2_n = np.sum(np.isfinite(g2), axis=1)
    # This table (unlike run_differential_analysis) covers every precursor, including ones with
    # zero observations in a group -- np.nanmean/nanstd correctly return NaN for those rows but
    # warn about it; the NaN is handled below (n<2 -> NaN CV, missingness -> ineligible), so the
    # warning is expected noise, not a bug.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        g1_mean = np.nanmean(g1, axis=1)
        g2_mean = np.nanmean(g2, axis=1)

        g1_linear = np.exp2(g1); g2_linear = np.exp2(g2)
        g1_linear_mean = np.nanmean(g1_linear, axis=1); g2_linear_mean = np.nanmean(g2_linear, axis=1)
        g1_sd = np.nanstd(g1_linear, axis=1, ddof=1); g2_sd = np.nanstd(g2_linear, axis=1, ddof=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        g1_cv = g1_sd / g1_linear_mean; g2_cv = g2_sd / g2_linear_mean
    g1_cv[(g1_n < 2) | (g1_linear_mean <= 0)] = np.nan
    g2_cv[(g2_n < 2) | (g2_linear_mean <= 0)] = np.nan

    g1_missing = 1.0 - g1_n / len(g1_samples)
    g2_missing = 1.0 - g2_n / len(g2_samples)
    eligible = (g1_missing <= max_missing_pct) & (g2_missing <= max_missing_pct)

    return pd.DataFrame({
        f"mean_log2_{group1_name}": g1_mean,
        f"mean_log2_{group2_name}": g2_mean,
        f"cv_{group1_name}": g1_cv,
        f"cv_{group2_name}": g2_cv,
        f"n_obs_{group1_name}": g1_n,
        f"n_obs_{group2_name}": g2_n,
        f"missing_pct_{group1_name}": g1_missing,
        f"missing_pct_{group2_name}": g2_missing,
        "DE_eligible": eligible,
    }, index=mat.index)


def compare_groups(mat, sample_to_group, group1, group2, max_missing_pct=0.5,
                    log_transform_input=True, pseudocount=None):
    """Run the full pipeline for a single group pair and return a results DataFrame."""
    g1_samples = get_group_samples(sample_to_group, group1)
    g2_samples = get_group_samples(sample_to_group, group2)
    print(f"{group1} vs {group2}: {len(g1_samples)} vs {len(g2_samples)} samples")

    pre_de = build_pre_de_table(
        mat, g1_samples, g2_samples, group1, group2,
        max_missing_pct=max_missing_pct,
        log_transform_input=log_transform_input,
        pseudocount=pseudocount,
    )

    filt_mat, g1_missing_pct, g2_missing_pct = filter_by_missingness(
        mat, g1_samples, g2_samples, max_missing_pct
    )

    log_mat = compute_log2(filt_mat, log_transform_input, pseudocount)

    results = run_differential_analysis(log_mat, g1_samples, g2_samples, group1, group2)

    #results[f"missing_pct_{group1}"] = g1_missing_pct.loc[results.index]
    #results[f"missing_pct_{group2}"] = g2_missing_pct.loc[results.index]

    return results, pre_de


def run_comparisons(mat, sample_to_group, group_pairs, max_missing_pct=0.5,
                    log_transform_input=True, pseudocount=None, return_pre_de=False):
    """Run compare_groups() for each (group1, group2) pair in group_pairs.

    Parameters
    ----------
    mat, sample_to_group : output of load_data()
    group_pairs : list of (group1, group2) tuples
    out_dir : if given, each comparison is also written to
        f"{out_dir}/{group1}_vs_{group2}.csv"

    Returns
    -------
    dict of {"{group1}_vs_{group2}": results DataFrame}
    """
    all_results = {}
    all_pre_de = {}
    for group1, group2 in group_pairs:
        key = f"{group1}_vs_{group2}"
        print(f"\n=== {key} ===")
        results, pre_de = compare_groups(
            mat, sample_to_group, group1, group2,
            max_missing_pct=max_missing_pct,
            log_transform_input=log_transform_input,
            pseudocount=pseudocount,
        )
        all_results[key] = results
        all_pre_de[key] = pre_de

    if return_pre_de:
        return all_results, all_pre_de
    return all_results


def classify_de_result(df, p_adj_cutoff=0.05, log2fc_cutoff=1.0,
                        padj_col="p_adj", log2fc_col="log2FC",species_col="Species", 
                        result_col="DE_result"):
    """
    Classify each row as a true positive, true negative, false positive, or
    false negative, based on the adjusted p-value, the measured log2 fold
    change, and a hard-coded expected trend per species:
 
        Human   -> expected stable        (no true DE)
        Yeast   -> expected upregulated   (log2FC > 0)
        E.coli  -> expected downregulated (log2FC < 0)
 
    This mirrors a fixed spike-in benchmark layout and may need to be
    adapted for other species/expected-direction combinations. Rows whose
    species is none of the three above are left unclassified (NA in
    `result_col`).
 
    Parameters
    ----------
    df : DataFrame containing at least [species_col, padj_col, log2fc_col]
    p_adj_cutoff : significance threshold on the adjusted p-value
    log2fc_cutoff : minimum |log2FC| required to call a change
 
    Returns
    -------
    DataFrame, a copy of df with `result_col` added/overwritten.
    """
    df = df.copy()
    df[result_col] = pd.Series(pd.NA, index=df.index, dtype="object")
 
    species = df[species_col]
    padj = df[padj_col]
    log2fc = df[log2fc_col]
 
    significant = padj < p_adj_cutoff
    large_change = log2fc.abs() > log2fc_cutoff
    not_de = padj.isna() | log2fc.isna() | (~significant) | (~large_change)
 
    # --- Human: expected stable (no true DE) ---
    is_human = species == "HUMAN"
    df.loc[is_human & not_de, result_col] = "true negative"
    df.loc[is_human & significant & large_change, result_col] = "false positive"
 
    # --- Yeast: expected upregulated ---
    is_yeast = species == "YEAST"
    df.loc[is_yeast & (log2fc > log2fc_cutoff) & significant, result_col] = "true positive"
    df.loc[is_yeast & (log2fc < -log2fc_cutoff) & significant, result_col] = "false positive"
    df.loc[is_yeast & not_de, result_col] = "false negative"
 
    # --- E.coli: expected downregulated ---
    is_ecoli = species == "ECOLI"
    df.loc[is_ecoli & (log2fc < -log2fc_cutoff) & significant, result_col] = "true positive"
    df.loc[is_ecoli & (log2fc > log2fc_cutoff) & significant, result_col] = "false positive"
    df.loc[is_ecoli & not_de, result_col] = "false negative"
 
    return df



# Global variables 
species_order = ["HUMAN", "YEAST", "ECOLI"]
color_human = "#199d76"
color_yeast = "#d85f02"
color_ecoli = "#7570b2"
palette = {
    "HUMAN": color_human, 
    "YEAST": color_yeast, 
    "ECOLI": color_ecoli
}

species_label = {'HUMAN': 'Human',
                 'YEAST': 'Yeast',
                 'ECOLI': 'E. coli'}

def clean_spines(
    ax,
    bottom=1, left=1, top=0, right=0,
    x_tick=1, y_tick=1,
    tick_labelsize=8,
    labelsize=9,
    legend_fontsize=8
):
    # Spine widths
    ax.spines['bottom'].set_linewidth(bottom)
    ax.spines['left'].set_linewidth(left)
    ax.spines['top'].set_linewidth(top)
    ax.spines['right'].set_linewidth(right)

    # Tick widths + label size
    ax.tick_params(axis='x', width=x_tick, labelsize=tick_labelsize)
    ax.tick_params(axis='y', width=y_tick, labelsize=tick_labelsize)

    # Axis label font size
    ax.xaxis.label.set_size(labelsize)
    ax.yaxis.label.set_size(labelsize)
    
    # title size
    ax.title.set_fontsize(labelsize)
    
    # Legend font size (if legend exists)
    legend = ax.get_legend()
    if legend is not None:
        for text in legend.get_texts():
            text.set_fontsize(legend_fontsize)


palette_blank = {
    "": "white", 
    "": "white", 
    "": "white", 
}

grey50 = "#7F7F7F"
FC_min = -3.2
FC_max = 3.2


def rasterize_scatter(ax):
    for coll in ax.collections:
        coll.set_rasterized(True)


def plot_scatter_box(df, title, x_col,
                     figsize, 
                     width_ratios, 
                     x_min_quantile=0.0005, 
                     x_max_ratio=1.02, 
                     show_yaxis=True, 
                     show_ylabel=True, 
                     show_yticks=True,
                     show_legend=True, 
                     show_counts=False,
                     show_sd=False):
    # ---------------------------
    # 1. Data Preparation
    # ---------------------------
    plot_df = df.copy()
    
    # Ensure Species is categorical for correct ordering in boxplot
    plot_df['Species'] = pd.Categorical(plot_df['Species'], categories=species_order, ordered=True)
    
    # Calculate log2 for scatter x-axis
    # Filter out non-positive values to avoid log errors if necessary
    #plot_df = plot_df[plot_df['ctr_mean'] > 0].copy()
    #plot_df['log2_ctr'] = np.log2(plot_df['ctr_mean'])

    # ---------------------------
    # 2. Figure Setup (Side-by-Side)
    # ---------------------------
    # Create figure with 2 subplots sharing the Y axis.
    # width_ratios=[4, 1] makes the scatter plot 4x wider than the box plot.
    fig, (ax_scatter, ax_box) = plt.subplots(
        nrows=1, ncols=2, 
        figsize=figsize, 
        sharey=True, 
        gridspec_kw={'width_ratios': width_ratios}
    )

    # ---------------------------
    # 3. Plotting: Left Panel (Scatter)
    # ---------------------------
    x_col = 'mean_log2_'+x_col
    sns.scatterplot(
        data=plot_df, x=x_col, y='log2FC', 
        hue='Species', palette=palette, markers='o',
        alpha=0.1, s=10, edgecolor=None, ax=ax_scatter, legend=False # Turn off default legend here
    )
    rasterize_scatter(ax_scatter)
    
    # Add Median Lines to Scatter
    for species in species_order:
        if species in plot_df['Species'].unique():
            species_data = plot_df.loc[plot_df['Species'] == species, 'log2FC']
            median_val = species_data.median()
            n_points = len(species_data)
            
            ax_scatter.axhline(y=median_val, color=palette[species], linestyle='--', 
                               alpha=0.9, linewidth=1, snap=True)
            
            if show_counts:
                # Add sample size text under median line
                ax_scatter.text(
                    ax_scatter.get_xlim()[1] - 5, median_val - 0.3,
                    f'n = {n_points}',
                    fontsize=8, color=palette[species], ha='left', va='top'
                )

    # Scatter Axes Styling
    ax_scatter.xaxis.set_major_locator(mticker.MaxNLocator(integer=True, nbins=6))
    ax_scatter.set_xlabel("$\mathregular{Log_2(B)}$", fontsize=11, color='black')
    ax_scatter.tick_params(axis='x', labelsize=10, width=0.3, color='black')
    ax_scatter.set_xlim(plot_df[x_col].quantile(x_min_quantile), plot_df[x_col].max()*x_max_ratio)
    
    ax_scatter.set_ylim(FC_min, FC_max)
    if show_yaxis:
        y_ticks = np.arange(round(FC_min), round(FC_max) + 1, 1)
        ax_scatter.set_yticks(y_ticks)
        if show_ylabel:
            ax_scatter.set_ylabel("$\mathregular{Log_2(A/B)}$", fontsize=11, color='black')
        else:
            ax_scatter.set_ylabel("")
        if show_yticks:
            ax_scatter.set_yticklabels(y_ticks)
        else:
            ax_scatter.set_yticklabels([])
    else:
        ax_scatter.set_yticks([])
        ax_scatter.set_ylabel("") 
        ax_scatter.spines['left'].set_visible(False)
    
    clean_spines(ax_scatter)
    
    # ---------------------------
    # 4. Plotting: Right Panel (Box Simple)
    # ---------------------------
    medianprops={"color": "black", "linewidth": 1}
    # Define outlier properties to match R script
    flierprops = dict(marker='o', markerfacecolor=grey50, markersize=2,
                      markeredgecolor='none', alpha=0.1)
    
    sns.boxplot(
        data=plot_df, x='Species', y='log2FC', hue='Species',
        palette=palette, dodge=False, ax=ax_box, linewidth=0.5,
        flierprops=flierprops, medianprops=medianprops,
        boxprops=dict(alpha=0.8)
    )

    # Box Axes Styling (Equivalent to theme_void())
    sns.despine(ax=ax_box, left=True, bottom=True, right=True, top=True)
    ax_box.set_xticks([]) # Remove x ticks
    # yticks are already hidden due to sharey=True, but we ensure no labels exist:
    ax_box.tick_params(left=False, labelleft=False) 
    ax_box.set_xlabel("")
    ax_box.set_ylabel("")
    if ax_box.get_legend() is not None:
        ax_box.get_legend().remove()
    
    # Add standard deviation labels on the right side of boxplot
    x_pos = ax_box.get_xlim()[1] + 0.15  # Position to the right of the box
    
    for idx, species in enumerate(species_order):
        if species in plot_df['Species'].unique():
            species_data = plot_df.loc[plot_df['Species'] == species, 'log2FC']
            median_val = species_data.median()
            if show_sd:
                sd_val = species_data.std()
                # Add SD text aligned with median line
                ax_box.text(
                    x_pos, median_val,
                    f'{sd_val:.2f}',
                    fontsize=9, color=palette[species], ha='left', va='center'
                )
    
    if show_sd:
        # Add 's.d.' label at the top
        ax_box.text(
            x_pos, ax_box.get_ylim()[1]*0.75,
            's.d.',
            fontsize=9, color='black', ha='left', va='bottom', fontweight='normal'
        )
    
    # ---------------------------
    # 5. Final Layout & Titles
    # ---------------------------
    # Title on the scatter plot axes
    ax_scatter.set_title(title, fontsize=9, fontweight='normal', pad=0, color='black')

    if show_legend:
        # Custom Legend with filled circles
        legend_handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, 
                                      markersize=5, label=species_label[label]) for label, color in palette.items()]
    else:
        legend_handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, 
                                      markersize=5, label=label) for label, color in palette_blank.items()]
    
    fig.legend(
        handles=legend_handles,
        loc='lower center',
        bbox_to_anchor=(0.4, 0.98), # Adjust horizontal position relative to scatter plot
        ncol=3, frameon=False, fontsize=8
    )
    
    # Tight layout, closer spacing between plots (wspace), room at top/bottom
    plt.subplots_adjust(top=0.92, bottom=0.15, wspace=0.02)
    
    return fig



def plot_volcano(ax, df, logfc_min, logfc_max, logfc_cutoff, p_cutoff, 
                 title, text_x, text_y, 
                 x_label = None, color_by='Species', colors=palette, 
                 show_TP_info = True, show_UpDown_info=False):
    # ---------------------------
    # 1. Data Preparation
    # ---------------------------
    plot_df = df.copy()
    
    # Ensure Species is categorical for correct ordering in boxplot
    plot_df['Species'] = pd.Categorical(plot_df['Species'], categories=species_order, ordered=True)
    plot_df['log_Padj'] = -np.log10(plot_df['p_adj'])
    
    # ---------------------------
    # 3. Plotting: Scatter
    # ---------------------------
    sns.scatterplot(
        data=plot_df, x='log2FC', y='log_Padj', 
        hue=color_by, palette=colors, markers='o',
        alpha=0.3, s=10, edgecolor=None, ax=ax, legend=False # Turn off default legend here
    )
    rasterize_scatter(ax)
    
    ax.axhline(y=-np.log10(p_cutoff), color='gray', linestyle='--', 
                       alpha=0.8, linewidth=1, snap=True)
    if logfc_cutoff:  # None/0: no fold-change cutoff in use, so no vertical threshold lines
        ax.axvline(x=logfc_cutoff, color='gray', linestyle='--',
                           alpha=0.8, linewidth=1, snap=True)
        ax.axvline(x=-logfc_cutoff, color='gray', linestyle='--',
                           alpha=0.8, linewidth=1, snap=True)

    # Scatter Axes Styling
    ax.set_xlim(logfc_min, logfc_max)
    #ax.set_ylim(-0.5, 15)
    x_ticks = np.arange(round(logfc_min), round(logfc_max)+1, 1)
    #y_ticks = (0, 5, 10)
    ax.set_xticks(x_ticks)
    #ax.set_yticks(y_ticks)
    ax.set_ylabel(r'$-\mathrm{Log}_{10}(\mathrm{Padj})$', fontsize=9, color='black')
    if x_label == None:
        ax.set_xlabel("$\\mathregular{Log_2(A/B)}$", fontsize=9, color='black')
    else:
        ax.set_xlabel("$\\mathregular{Log_2FC}$", fontsize=9, color='black')
    ax.tick_params(axis='both', labelsize=8, width=0.3, color='black')
    clean_spines(ax)

    # ---------------------------
    # 5. Final Layout & Titles
    # ---------------------------
    # Title on the scatter plot axes
    ax.set_title(title, fontsize=9, fontweight='normal', pad=0, color='black')
    
    # show TP info
    if show_TP_info:
        pos_neg_counts = plot_df['DE_result'].value_counts()
        n_fp, n_tp = pos_neg_counts.get('false positive', 0), pos_neg_counts.get('true positive', 0)
        # Strict FDR (as performance.metric_rows): wrong-direction calls count as false discoveries.
        n_wd = pos_neg_counts.get('wrong direction', 0)
        fdr = 100 * (n_fp + n_wd) / (n_fp + n_wd + n_tp) if (n_fp + n_wd + n_tp) else float('nan')
        if text_y is None:
            final_text_y = ax.get_ylim()[1] * 0.95
        else:
            final_text_y = text_y
        n_fn = pos_neg_counts.get('false negative', 0)
        # Sensitivity as performance.metric_rows: TP / (TP + FN + wrong direction)
        sens = 100 * n_tp / (n_tp + n_fn + n_wd) if (n_tp + n_fn + n_wd) else float('nan')
        ax.text(text_x, final_text_y, f"TP: {n_tp}\nTN: {pos_neg_counts.get('true negative', 0)}\nFP: {n_fp}\nFN: {n_fn}\nWD: {n_wd}\n\nSensitivity: {sens:.2f}%\nEmpirical FDR: {fdr:.2f}%",
                        va='top', fontsize=8)    
    
    # show Up-down info
    if show_UpDown_info:
        up_down_counts = plot_df['UpDown'].value_counts()
        if text_y is None:
            final_text_y = ax.get_ylim()[1] * 0.95
        else:
            final_text_y = text_y
        ax.text(logfc_min*0.95, final_text_y, f"Down: {up_down_counts['Down']}", 
                        va='top', fontsize=8)  
        ax.text(logfc_max*0.35, final_text_y, f"Up: {up_down_counts['Up']}",
                        va='top', fontsize=8)

    return ax


# benchmark_state codes (performance.assign_benchmark_state) -> plot_volcano's DE_result labels.
# "wrong_direction" (significant, but opposite the expected sign) is not a TP, and counts as a false
# discovery in performance.metric_rows' (strict) FDR, so it gets its own label here -- plot_volcano's
# the empirical FDR then matches D's numerator/denominator exactly.
_BENCHMARK_STATE_LABEL = {"TP": "true positive", "FP": "false positive", "FN": "false negative",
                          "TN": "true negative", "wrong_direction": "wrong direction"}


def plot_benchmark_volcanos(m, stratum_value, methods, p_cutoff, logfc_cutoff=None,
                            stratum_col="completeness_state", figsize=(5.2, 2.6), title_prefix=""):
    """One volcano per method, drawn from EXACTLY the rows performance.compute_sensitivity_fdr
    uses for panel D's sensitivity/FDR: `m` (prepare_pair's merged DIA-NN x JUMPlion table)
    filtered to `m[stratum_col] == stratum_value` (the partially-complete stratum), then each
    method's own log2FC_<suf>/p_adj_<suf>/Species_std_<suf>/benchmark_state_<suf> columns, with
    benchmark_state == "NE" dropped (as metric_rows does). `methods`: [(display name, suffix), ...]
    -- e.g. [("DIA-NN", "D"), ("JUMPlion", "J")]. The in-axes text is plot_volcano's own TP/TN/
    FP/FN counts plus sensitivity and empirical FDR, so it reproduces D's empirical FDR for that method. Both subplots share
    x/y limits (so the two methods read on the same scale) -- x symmetric around 0, y from 0.
    Returns the Figure; the caller saves it."""
    frames = _benchmark_volcano_frames(m, stratum_value, methods, stratum_col)
    fc_lim, y_max = _volcano_limits([df for _, df in frames])
    fig, axs = plt.subplots(1, len(frames), figsize=figsize, sharex=True, sharey=True, squeeze=False)
    for ax, (name, df) in zip(axs[0], frames):
        ax.set_ylim(0, y_max * 1.05)  # before plot_volcano, whose count text is placed from ylim
        plot_volcano(ax, df, -fc_lim, fc_lim, logfc_cutoff, p_cutoff, f"{title_prefix}{name}",
                     text_x=-fc_lim * 0.95, text_y=None, x_label="log2FC")
    for ax in axs[0][1:]:
        ax.set_ylabel("")
    fig.tight_layout()
    return fig


def _benchmark_volcano_frames(m, stratum_value, methods, stratum_col="completeness_state"):
    """[(method display name, volcano DataFrame), ...] -- see plot_benchmark_volcanos."""
    g = m[m[stratum_col] == stratum_value]
    frames = []
    for name, suf in methods:
        df = pd.DataFrame({"log2FC": g[f"log2FC_{suf}"], "p_adj": g[f"p_adj_{suf}"],
                           "Species": g[f"Species_std_{suf}"],
                           "state": g[f"benchmark_state_{suf}"]})
        df = df[df["state"] != "NE"].copy()
        df["DE_result"] = df["state"].map(_BENCHMARK_STATE_LABEL)
        frames.append((name, df))
    return frames


def _volcano_limits(dfs):
    """(symmetric |log2FC| limit, max -log10(p_adj)) across every DataFrame in `dfs`."""
    allp = pd.concat(dfs)
    fc_lim = max(2.0, float(np.ceil(allp["log2FC"].abs().quantile(0.999) * 2) / 2)) if len(allp) else 2.0
    y_max = float((-np.log10(allp["p_adj"].clip(lower=1e-300))).max()) if allp["p_adj"].notna().any() else 1.0
    return fc_lim, y_max


def plot_benchmark_volcano_grid(entries, methods, stratum_value, p_cutoff, logfc_cutoff=None,
                                datasets_per_row=2, panel_size=(2.4, 2.4)):
    """Every dataset's benchmark volcanos (see plot_benchmark_volcanos) in ONE figure.
    `entries`: [(dataset display label, merged table m), ...] in display order; each dataset takes
    len(methods) adjacent columns, `datasets_per_row` datasets per row (e.g. Astral 5/15 min on row
    1, timsTOF 5/15 min on row 2). All panels share ONE x and y range, computed across every
    dataset, so they read on the same scale. Returns the Figure; the caller saves it."""
    per = [(label, _benchmark_volcano_frames(m, stratum_value, methods)) for label, m in entries]
    fc_lim, y_max = _volcano_limits([df for _, fr in per for _, df in fr])
    nrows = -(-len(per) // datasets_per_row)
    ncols = datasets_per_row * len(methods)
    fig, axs = plt.subplots(nrows, ncols, figsize=(panel_size[0] * ncols, panel_size[1] * nrows),
                            sharex=True, sharey=True, squeeze=False)
    for i, (label, frames) in enumerate(per):
        r, c0 = divmod(i, datasets_per_row)
        for j, (name, df) in enumerate(frames):
            ax = axs[r, c0 * len(methods) + j]
            ax.set_ylim(0, y_max * 1.05)
            plot_volcano(ax, df, -fc_lim, fc_lim, logfc_cutoff, p_cutoff, f"{label}\n{name}",
                         text_x=-fc_lim * 0.95, text_y=None, x_label="log2FC")
    for ax in axs.ravel()[len(per) * len(methods):]:
        ax.axis("off")
    for r in range(nrows):
        for ax in axs[r, 1:]:
            ax.set_ylabel("")
    fig.tight_layout()
    return fig
