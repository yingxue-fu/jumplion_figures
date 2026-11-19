"""Supp_Benchmark_design: the HYE-Wang design (Supp_Performance_A_benchmark_design, from the hye_wang
run) stacked above the generalization design (Supp_Generalization_benchmark_design, from the protein-level
hye_genbeta run). The two panels come from different runs, so this tiles their saved PNGs -- run both
targets first ("png" must be in style.formats). Written next to the performance supp panels.

    python -m jumplion_figures.design_figure --top configs/hye_wang.json --bottom configs/hye_genbeta_protein.json
"""
import argparse
import logging
import sys
from pathlib import Path

import matplotlib
import numpy as np
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from . import style
from .common import PathResolver, read_raw_config, substitute_config

log = logging.getLogger("jumplion_figures")

TOP_STEM = "Supp_Performance_A_benchmark_design"
BOTTOM_STEM = "Supp_Generalization_benchmark_design"
OUT_STEM = "Supp_Benchmark_design"


def _config(path, output_dir):
    raw = read_raw_config(path)
    cfg = substitute_config(raw, strict=not raw.get("datasets"))
    root = Path(output_dir) if output_dir else PathResolver(cfg["_config_dir"])(cfg.get("output_dir", "jumplion_figures_output"))
    return cfg, root


def _trim_rows(img, dpi, pad=0.04):
    """Drop the empty rows above and below the content (the panels are saved at their exact size, so
    they carry blank margins). Columns are untouched, so the two panels stay horizontally aligned."""
    ink = (img[..., :3] < 0.98).any(axis=2)
    if img.shape[2] == 4:
        ink &= img[..., 3] > 0.01
    rows = np.flatnonzero(ink.any(axis=1))
    if rows.size == 0:
        return img
    p = int(round(pad * dpi))
    return img[max(rows[0] - p, 0):rows[-1] + 1 + p]


def combine(top_cfg, bottom_cfg, output_dir=None, gap=0.03, label_h=0.15):
    top, top_root = _config(top_cfg, output_dir)
    _, bottom_root = _config(bottom_cfg, output_dir)
    paths = {"a": top_root / "performance" / "supp" / f"{TOP_STEM}.png",
             "b": bottom_root / "generalization" / "supp" / f"{BOTTOM_STEM}.png"}
    missing = [str(p) for p in paths.values() if not p.exists()]
    if missing:
        log.warning("%s skipped: missing panel PNG(s) %s (run hye_wang and hye_genbeta first; "
                    "\"png\" must be in style.formats)", OUT_STEM, missing)
        return 1
    style_cfg = top.get("style", {})
    style.apply_style(style_cfg)
    saver = style.FigureSaver(style_cfg.get("formats", ["pdf", "png"]), style_cfg.get("dpi", 600))
    imgs = {k: _trim_rows(plt.imread(p), saver.dpi) for k, p in paths.items()}
    w ={k: img.shape[1] / saver.dpi for k, img in imgs.items()}
    h = {k: img.shape[0] / saver.dpi for k, img in imgs.items()}
    # both rows keep their native size (so the bars stay the same width), left-aligned
    W = max(w.values())
    ph = h
    total_h = sum(ph.values()) + 2 * label_h + gap
    fig = plt.figure(figsize=(W, total_h))
    y = total_h
    for k in ("a", "b"):
        # panel letter in its own strip above the row, clear of the bars and sample labels
        fig.text(0.06 / W, (y - 0.04) / total_h, k, fontsize=9, fontweight="bold", ha="left", va="top")
        y -= label_h + ph[k]
        ax = fig.add_axes([0, y / total_h, w[k] / W, ph[k] / total_h])
        ax.imshow(imgs[k])
        ax.axis("off")
        y -= gap
    out = top_root / "performance" / "supp"
    saver.save(fig, out, OUT_STEM, tight=False)
    log.info("wrote %s", out / OUT_STEM)
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--top", required=True, help="config of the hye_wang run")
    ap.add_argument("--bottom", required=True, help="config of the protein-level hye_genbeta run")
    ap.add_argument("--output-dir", help="override output_dir of both configs")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    return combine(args.top, args.bottom, args.output_dir)


if __name__ == "__main__":
    sys.exit(main())
