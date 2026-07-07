"""
shap_rtl.plots.summary  –  RTL-aware summary plot.

Wraps shap.summary_plot and post-processes the y-axis tick labels so that
Urdu/Arabic/Persian feature names are rendered via HarfBuzz+FreeType images
instead of matplotlib text glyphs.

Strategy
--------
Call shap.summary_plot(show=False), then apply RTL rendering to whichever
y-tick labels contain RTL codepoints. Axes geometry, colours, colourbar, and
sorting are fully delegated to the original function.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from ..rtl_utils import (
    _find_rtl_font,
    is_rtl_text,
    apply_rtl_ytick_images,
    hide_rtl_text_ticks,
)


# ── shared y-tick image injector (mirrors bar/beeswarm pattern) ───────────────
def _inject_image_label(
    ax, fig, text: str, ydata: float, font_path: str,
    color: str = "black", font_size_pt: float = 12,
):
    from matplotlib.offsetbox import AnnotationBbox, OffsetImage
    from ..rtl_utils import render_rtl_label, render_mixed_label
    import matplotlib.colors as mcolors

    # "0.123 = عمر" → mixed rendering; pure RTL → rtl rendering
    if " = " in text and is_rtl_text(text.split(" = ", 1)[-1]):
        img_pil = render_mixed_label(text, font_path, font_size_pt=font_size_pt, dpi=150)
    else:
        img_pil = render_rtl_label(text, font_path, font_size_pt=font_size_pt, dpi=150)

    if img_pil.width < 2:
        return

    arr = np.array(img_pil, dtype=np.uint8).copy()
    try:
        rgb = mcolors.to_rgb(color)
        arr[:, :, 0] = int(rgb[0] * 255)
        arr[:, :, 1] = int(rgb[1] * 255)
        arr[:, :, 2] = int(rgb[2] * 255)
    except Exception:
        pass

    oi = OffsetImage(arr, zoom=72.0 / 150.0)
    oi.image.axes = ax
    ab = AnnotationBbox(
        oi,
        xy=(0, ydata),
        xycoords=("axes fraction", "data"),
        xybox=(-4, 0),
        boxcoords="offset points",
        frameon=False,
        box_alignment=(1.0, 0.5),
        pad=0,
    )
    ax.add_artist(ab)


def _get_main_ax(fig) -> plt.Axes:
    """Return the main plot axes (the one with y-tick labels)."""
    axes = fig.get_axes()
    if not axes:
        return plt.gca()
    # Pick the axes that has the most populated y-tick labels
    for ax in axes:
        labels = [t.get_text() for t in ax.get_yticklabels()]
        if any(labels):
            return ax
    return axes[0]


# ── public API ─────────────────────────────────────────────────────────────────
def summary_plot(
    shap_values,
    features=None,
    feature_names=None,
    max_display: int = 10,
    plot_type=None,
    color=None,
    axis_color: str = "#333333",
    title=None,
    alpha: float = 1,
    show: bool = True,
    sort: bool = True,
    color_bar: bool = True,
    plot_size="auto",
    **kwargs,
):
    """
    RTL-aware summary plot (drop-in for shap.summary_plot).

    Identical to shap.summary_plot except that feature names containing
    Urdu/Arabic/Persian characters are rendered via HarfBuzz+FreeType images.

    Parameters
    ----------
    shap_values : np.ndarray or shap.Explanation
        Matrix of SHAP values (shape n_samples × n_features) or an Explanation.
    features : np.ndarray or pd.DataFrame, optional
        Feature values for colouring dots (beeswarm / dot type).
    feature_names : list of str, optional
        Feature names. RTL names are auto-detected.
    max_display : int
        Maximum features to show.
    plot_type : {"dot","beeswarm","bar","violin","compact_dot"}, optional
        Plot style. Defaults to "dot" (beeswarm) for Explanation objects.
    color : colour, optional
    axis_color : str
    title : str, optional
    alpha : float
    show : bool
        If False, returns the main Axes instead of calling plt.show().
    sort : bool
    color_bar : bool
    plot_size : "auto" | float | tuple
    **kwargs
        Forwarded verbatim to shap.summary_plot (e.g. layered_violin_max_num_bins,
        class_names, cmap, …).
    """
    import shap as _shap

    _shap.summary_plot(
        shap_values,
        features=features,
        feature_names=feature_names,
        max_display=max_display,
        plot_type=plot_type,
        color=color,
        axis_color=axis_color,
        title=title,
        alpha=alpha,
        show=False,
        sort=sort,
        color_bar=color_bar,
        plot_size=plot_size,
        **kwargs,
    )

    fig = plt.gcf()
    ax = _get_main_ax(fig)
    fig.canvas.draw()

    tick_labels = [t.get_text() for t in ax.get_yticklabels()]
    rtl_mode = any(is_rtl_text(lbl) for lbl in tick_labels)

    if rtl_mode:
        font_path, _ = _find_rtl_font()
        if font_path is not None:
            hide_rtl_text_ticks(ax, tick_labels)
            apply_rtl_ytick_images(
                ax, tick_labels, font_path,
                font_size_pt=12, color=axis_color,
            )

    if show:
        plt.show()
    else:
        return ax
