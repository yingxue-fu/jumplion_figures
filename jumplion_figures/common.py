"""Shared infrastructure: config loading, path resolution, output layout, table I/O."""
from __future__ import annotations

import json
import logging
import re
from copy import deepcopy
from pathlib import Path

import pandas as pd

log = logging.getLogger("jumplion_figures")


def setup_logging(level=logging.INFO):
    logging.basicConfig(level=level, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
                        datefmt="%H:%M:%S")
    for name in ("matplotlib", "PIL", "fontTools", "fontTools.subset"):
        logging.getLogger(name).setLevel(logging.WARNING)


_VAR = re.compile(r"\$\{([A-Za-z0-9_]+)\}")


def _substitute(obj, variables, strict=True):
    if isinstance(obj, str):
        def repl(m):
            if m.group(1) not in variables:
                if not strict:
                    return m.group(0)  # leave ${name} as-is; caller only wants a best-effort pass
                raise KeyError(f"Undefined config variable ${{{m.group(1)}}}; define it under \"paths\".")
            return variables[m.group(1)]
        return _VAR.sub(repl, obj)
    if isinstance(obj, list):
        return [_substitute(v, variables, strict) for v in obj]
    if isinstance(obj, dict):
        return {k: _substitute(v, variables, strict) for k, v in obj.items()}
    return obj


def read_raw_config(path):
    """Load the JSON config as-is (no ${name} expansion yet)."""
    path = Path(path).resolve()
    cfg = json.loads(path.read_text(encoding="utf-8"))
    cfg["_config_dir"] = str(path.parent)
    return cfg


def substitute_config(cfg, path_overrides=None, strict=True):
    """Expand ${name} using the top-level "paths" block (which may refer to itself).

    `path_overrides` lets a caller run the same raw config through substitution more than once
    with different "paths" values -- e.g. one pass per dataset when "datasets" supplies a
    different jumplion_outdir each time (see make_figures.py's per-dataset loop).

    `strict=False` leaves an undefined ${name} as-is instead of raising -- for a "base" pass over
    a config whose "datasets" block supplies jumplion_outdir (and so ${jumplion_outdir}-derived paths) per
    dataset rather than at the top level: that pass only needs the handful of truly top-level
    keys (style, output_dir, run), so it shouldn't fail just because dataset-scoped variables
    aren't defined yet.
    """
    cfg = deepcopy(cfg)
    if path_overrides:
        cfg.setdefault("paths", {}).update(path_overrides)
    variables = {k: str(v) for k, v in cfg.get("paths", {}).items()}
    for _ in range(10):  # resolve nested references like "${jumplion_outdir}/publication"
        new = {k: _substitute(v, variables, strict) for k, v in variables.items()}
        if new == variables:
            break
        variables = new
    return _substitute(cfg, variables, strict)


def load_config(path):
    """Load JSON config; expand ${name} using the top-level "paths" block (which may refer to itself)."""
    return substitute_config(read_raw_config(path))


class PathResolver:
    """Resolve config paths; relative paths are taken relative to the config file."""

    def __init__(self, base):
        self.base = Path(base)

    def __call__(self, p):
        if p is None or p == "":
            return None
        q = Path(p)
        return q if q.is_absolute() else self.base / q


class OutputLayout:
    """<output_dir>/<module>/{main,supp,table}, or <output_dir>/<module>/<dataset>/{main,supp,table}
    when `dataset` is given -- grouped by stage first, dataset second, so e.g. every dataset's
    differential_expression output lives under one differential_expression/ folder."""

    def __init__(self, root, name, dataset=None):
        self.root = Path(root) / name / dataset if dataset else Path(root) / name
        self.main = self.root / "main"
        self.supp = self.root / "supp"
        self.table = self.root / "table"
        for d in (self.main, self.supp, self.table):
            d.mkdir(parents=True, exist_ok=True)


def read_table(path, columns=None):
    """Read TSV/CSV/parquet. For parquet, only `columns` present in the file are loaded."""
    if path is None:
        return None
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    suffix = path.suffix.lower()
    if suffix in {".parquet", ".pq"}:
        if columns is not None:
            import pyarrow.parquet as pq
            names = pq.ParquetFile(path).schema_arrow.names
            keep = [c for c in names if c in set(columns)]
            return pd.read_parquet(path, columns=keep)
        return pd.read_parquet(path)
    sep = "," if suffix == ".csv" else "\t"
    df = pd.read_csv(path, sep=sep)
    if columns is not None:
        df = df[[c for c in df.columns if c in set(columns)]]
    return df


def write_tsv(df, directory, name, index=False):
    """Write a TSV (gzip if name ends with .gz) into a table folder."""
    path = Path(directory) / name
    kw = {"compression": "gzip"} if name.endswith(".gz") else {}
    df.to_csv(path, sep="\t", index=index, **kw)
    return path
