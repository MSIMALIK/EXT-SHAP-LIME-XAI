"""
shap_rtl.plots.force – RTL-aware force (additive) plot.

Leverages the original shap.plots.force core module for robust data tracking,
then post-intercepts the text elements to inject HarfBuzz-rendered labels
displaying true live model metrics.
"""

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import shap

from ..rtl_utils import _find_rtl_font, is_rtl_text, render_rtl_label


def _text_as_image(ax, x, y, text, font_path, fontsize=12, color="#FF0D57", ha="right"):
    """Places a HarfBuzz-rendered image of the text at axes data coordinates."""
    from matplotlib.offsetbox import AnnotationBbox, OffsetImage
    import matplotlib.colors as mcolors

    img_pil = render_rtl_label(text, font_path, font_size_pt=fontsize, dpi=150)
    if img_pil.width < 2:
        return None

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

    box_align = (1.0, 0.5) if ha == "right" else (0.0, 0.5)

    ab = AnnotationBbox(
        oi,
        xy=(x, y),
        xycoords="data",
        frameon=False,
        box_alignment=box_align,
        pad=0,
    )
    ax.add_artist(ab)
    return ab


def force(shap_values, figsize=(20, 3), show=True, text_rotation=0, min_perc=0.05):
    """
    RTL-aware force plot using the original SHAP module directly.
    Extracts live model feature inputs and actual SHAP metrics directly from the graph.
    """
    # 1. Turn off immediate rendering so we can process the figure elements safely
    plt.ioff()

    # 2. Extract configuration from the Explanation object if present to detect RTL text
    feature_names = getattr(shap_values, "feature_names", None)
    
    rtl_mode = False
    if feature_names is not None:
        rtl_mode = any(is_rtl_text(str(n)) for n in feature_names)
    
    font_path, _ = _find_rtl_font()

    # 3. Call the ORIGINAL native SHAP force plot module directly.
    # This guarantees that the real feature names, model metrics, and internal tracking are preserved perfectly.
    fig = shap.plots.force(
        shap_values,
        matplotlib=True,
        figsize=figsize,
        show=False,
        text_rotation=text_rotation,
        contribution_threshold=min_perc
    )
    
    if fig is None:
        fig = plt.gcf()
    ax = fig.gca()

    # 4. Post-process the generated text elements
    if font_path is not None:
        fig.canvas.draw()
        
        # Capture all raw text labels generated below the bars by original SHAP
        all_texts = list(ax.texts)
        
        for txt in all_texts:
            label_string = txt.get_text()
            
            # Skip empty strings, or layout controls like 'higher', 'lower', or 'f(x)'
            if not label_string or label_string in ["higher", "lower", "f(x)"] or "=" not in label_string:
                continue
                
            # Original SHAP text is structured as: "model_feature_value = feature_name"
            parts = label_string.split("=", 1)
            feat_val_part = parts[0].strip()   # This is the actual feature value from the model pipeline
            feat_name_part = parts[1].strip()  # This is the string label or token word
            
            # Check if this feature name belongs to an RTL script language sequence
            if is_rtl_text(feat_name_part):
                x_pos, y_pos = txt.get_position()
                ha_align = txt.get_transform_radial() if hasattr(txt, 'get_transform_radial') else txt.get_ha()
                txt_color = txt.get_color()
                
                # Format dynamically to show the feature word and the real feature value side by side
                clean_rtl_label = f"{feat_name_part} ({feat_val_part})"
                
                # Turn off the native text element entirely to hide broken LTR square bracket artifacts
                txt.set_text("")
                txt.set_visible(False)
                
                # Inject the clean HarfBuzz script image at the precise native text anchor position
                _text_as_image(
                    ax=ax,
                    x=x_pos,
                    y=y_pos,
                    text=clean_rtl_label,
                    font_path=font_path,
                    fontsize=12,
                    color=txt_color,
                    ha=ha_align
                )

    if show:
        plt.show()
    else:
        return fig