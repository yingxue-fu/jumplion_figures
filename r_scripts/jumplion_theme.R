# ggplot2 theme matching the Python figures (jumplion_figures/style.py apply_style, font_size = 8):
# Arial, axis labels / strips 8 pt, tick labels and legend 7 pt, titles 9 pt bold, 0.8 pt axes, 3 pt ticks,
# no top / right spines, no legend frame. Sizes of geom_text() are in mm: use pt_mm().
FS <- 8
pt_mm <- function(pt) pt / ggplot2::.pt
LW_AXIS <- 0.8 / (0.75 * ggplot2::.pt)   # 0.8 pt in ggplot linewidth units
LW_PATCH <- 0.5 / (0.75 * ggplot2::.pt)  # bar edges, 0.5 pt
LW_LINE <- 1.3 / (0.75 * ggplot2::.pt)   # curves, 1.3 pt
# Arial if installed, else a metric-compatible substitute (same glyph widths, so the layout does not change),
# else the default sans font. Say so, since the figures then differ from the Python ones in typeface only.
pick_font <- function() {
  avail <- tryCatch(unique(systemfonts::system_fonts()$family), error = function(e) character())
  for (f in c("Arial", "Liberation Sans", "Arimo", "Helvetica", "Nimbus Sans")) if (f %in% avail) return(f)
  "sans"
}
JL_FONT <- pick_font()
if (JL_FONT != "Arial") message("Arial not found; using '", JL_FONT, "' (see README: Fonts)")
theme_jumplion <- function(fs = FS) {
  ggplot2::theme_classic(base_size = fs, base_family = JL_FONT) +
    ggplot2::theme(
      text = ggplot2::element_text(family = JL_FONT, size = fs, colour = "black"),
      axis.text = ggplot2::element_text(size = fs - 1, colour = "black"),
      axis.title = ggplot2::element_text(size = fs),
      axis.line = ggplot2::element_line(linewidth = LW_AXIS),
      axis.ticks = ggplot2::element_line(linewidth = LW_AXIS),
      axis.ticks.length = grid::unit(3, "pt"),
      strip.background = ggplot2::element_blank(), strip.text = ggplot2::element_text(size = fs),
      strip.placement = "outside",
      plot.title = ggplot2::element_text(size = fs + 1, face = "bold", hjust = 0),
      plot.tag = ggplot2::element_text(size = fs + 1, face = "bold"),
      legend.text = ggplot2::element_text(size = fs - 1), legend.title = ggplot2::element_text(size = fs - 1),
      legend.key.size = grid::unit(0.35, "cm"), legend.margin = ggplot2::margin(0, 0, 0, 0),
      legend.background = ggplot2::element_blank())
}
save_fig <- function(stem, p, width, height) {  # PNG at 600 dpi + PDF with embedded font
  ggplot2::ggsave(paste0(stem, ".png"), p, width = width, height = height, dpi = 600)
  ggplot2::ggsave(paste0(stem, ".pdf"), p, width = width, height = height,
                  device = if (capabilities("cairo")) grDevices::cairo_pdf else grDevices::pdf)
}
