"""
shap_rtl.rtl_utils
──────────────────
RTL text utilities with proper language-specific font handling.
Supports: Arabic, Urdu, Persian, Hebrew
"""

import os
import warnings
import numpy as np
from PIL import Image

# ── font location ──────────────────────────────────────────────────────────────
_PKG_DIR = os.path.dirname(os.path.abspath(__file__))
_FONTS_DIR = os.path.join(os.path.dirname(_PKG_DIR), "fonts")

# ============================================================================
# FONT CONFIGURATION - Language-specific fonts
# ============================================================================

_LANGUAGE_FONTS = {
    "Arabic": [
        "NotoSansArabic-Regular.ttf",
        "NotoSansArabic-VariableFont_wdth,wght.ttf",
        "Amiri-Regular.ttf",
        "Amiri-Bold.ttf",
        "ScheherazadeNew-Regular.ttf",
        "Scheherazade-Regular.ttf",
        "Lateef-Regular.ttf",
        "Harmattan-Regular.ttf",
        "DroidNaskh-Regular.ttf",
    ],
    "Urdu": [
        "JameelNooriNastaliq.ttf",
        "JameelNooriNastaleeq.ttf",
        "Jameel Noori Nastaleeq Regular.ttf",
        "NotoNastaliqUrdu-Regular.ttf",
        "NotoNastaliqUrdu-VariableFont_wght.ttf",
        "UrduNastaliq.ttf",
        "NafeesNastaleeq.ttf",
    ],
    "Persian": [
        "NotoSansArabic-Regular.ttf",
        "NotoNaskhArabic-Regular.ttf",
        "Amiri-Regular.ttf",
        "ScheherazadeNew-Regular.ttf",
        "Vazir.ttf",
        "Vazirmatn-Regular.ttf",
    ],
    "Hebrew": [
        "NotoSansHebrew-Regular.ttf",
        "NotoSansHebrew-VariableFont_wdth,wght.ttf",
        "NotoSerifHebrew-Regular.ttf",
        "NotoSerifHebrew-VariableFont_wdth,wght.ttf",
        "Arial.ttf",
        "TimesNewRoman.ttf",
        "EzraSIL-Regular.ttf",
        "SBLHebrew-Regular.ttf",
    ],
    "Pashto": [
        "NotoSansArabic-Regular.ttf",
        "NotoNaskhArabic-Regular.ttf",
        "Amiri-Regular.ttf",
        "ScheherazadeNew-Regular.ttf",
        "JameelNooriNastaliq.ttf",
    ],
    "Dari": [
        "NotoSansArabic-Regular.ttf",
        "Amiri-Regular.ttf",
        "ScheherazadeNew-Regular.ttf",
        "JameelNooriNastaliq.ttf",
    ],
    "Kurdish": [
        "NotoSansArabic-Regular.ttf",
        "Amiri-Regular.ttf",
        "ScheherazadeNew-Regular.ttf",
        "NotoSansHebrew-Regular.ttf",
    ],
}

# ============================================================================
# FONT KEYWORDS FOR DETECTION (PRIORITIZED)
# ============================================================================

_LANGUAGE_KEYWORDS = {
    "Hebrew": [
        "hebrew",
        "ezra",
        "sbl",
        "noto sans hebrew",
        "noto serif hebrew",
    ],
    "Arabic": [
        "arabic",
        "naskh",
        "amiri",
        "scheherazade",
        "lateef",
        "harmattan",
        "droidnaskh",
    ],
    "Urdu": [
        "nastaliq",
        "nastaleeq",
        "jameel",
        "nafees",
        "urdu",
        "notonastaliqurdu",
    ],
    "Persian": [
        "persian",
        "farsi",
        "vazir",
        "vazirmatn",
    ],
}

_EXCLUDED_FONTS = {
    "Hebrew": ["arabic", "nastaliq", "urdu", "persian", "naskh", "amiri", "scheherazade", "lateef", "harmattan"],
    "Arabic": ["hebrew", "nastaliq", "urdu", "persian"],
    "Urdu": ["hebrew", "arabic", "persian"],
    "Persian": ["hebrew", "urdu", "arabic"],
}

# ============================================================================
# GLOBAL STATE - Track current language
# ============================================================================

_CURRENT_LANGUAGE = None
_RTL_FONT_PATH = None
_FONT_CACHE = {}
_FONT_REGISTERED = False

def set_current_language(language: str):
    """Set the current language for font detection."""
    global _CURRENT_LANGUAGE
    _CURRENT_LANGUAGE = language
    print(f"[DEBUG] Current language set to: {language}")


def get_current_language() -> str:
    """Get the current language."""
    return _CURRENT_LANGUAGE


# ============================================================================
# FONT DOWNLOADER - AUTO DOWNLOAD IF MISSING
# ============================================================================

def _download_font(language: str) -> str | None:
    """
    Auto-download font if not available.
    """
    fonts_dir = _FONTS_DIR
    os.makedirs(fonts_dir, exist_ok=True)
    
    font_map = {
        "Hebrew": {
            "url": "https://github.com/google/fonts/raw/main/ofl/notosanshebrew/NotoSansHebrew%5Bwdth,wght%5D.ttf",
            "filename": "NotoSansHebrew-Regular.ttf"
        },
        "Arabic": {
            "url": "https://github.com/google/fonts/raw/main/ofl/notosansarabic/NotoSansArabic%5Bwdth,wght%5D.ttf",
            "filename": "NotoSansArabic-Regular.ttf"
        },
        "Urdu": {
            "url": "https://github.com/google/fonts/raw/main/ofl/notonastaliqurdu/NotoNastaliqUrdu%5Bwght%5D.ttf",
            "filename": "NotoNastaliqUrdu-Regular.ttf"
        },
        "Persian": {
            "url": "https://github.com/google/fonts/raw/main/ofl/notosansarabic/NotoSansArabic%5Bwdth,wght%5D.ttf",
            "filename": "NotoSansArabic-Regular.ttf"
        },
    }
    
    if language not in font_map:
        return None
    
    font_info = font_map[language]
    font_path = os.path.join(fonts_dir, font_info["filename"])
    
    if os.path.exists(font_path):
        return font_path
    
    try:
        import urllib.request
        print(f"[INFO] Downloading {language} font...")
        urllib.request.urlretrieve(font_info["url"], font_path)
        print(f"[INFO] Font downloaded: {font_path}")
        return font_path
    except Exception as e:
        print(f"[WARNING] Could not download font: {e}")
        return None


# ============================================================================
# FONT MANAGEMENT - FIXED FOR ALL LANGUAGES
# ============================================================================

def _find_rtl_font(language: str = None) -> tuple[str | None, str]:
    """
    Find the best font for the specified language.
    Supports: Arabic, Urdu, Persian, Hebrew
    """
    global _RTL_FONT_PATH, _CURRENT_LANGUAGE
    
    # If no language specified, use the current language
    if language is None:
        language = _CURRENT_LANGUAGE
    
    # If still no language, default to Arabic
    if language is None:
        language = "Arabic"
    
    # Check cache first
    cache_key = language
    if cache_key in _FONT_CACHE:
        cached_path, cached_name = _FONT_CACHE[cache_key]
        if language in ["Urdu", "Hebrew", "Persian"] and cached_name:
            is_wrong_font = False
            if language == "Urdu" and ("arabic" in cached_name.lower() or "hebrew" in cached_name.lower()):
                is_wrong_font = True
            elif language == "Hebrew" and ("arabic" in cached_name.lower() or "nastaliq" in cached_name.lower()):
                is_wrong_font = True
            elif language == "Persian" and ("hebrew" in cached_name.lower() or "nastaliq" in cached_name.lower()):
                is_wrong_font = True
            
            if is_wrong_font:
                del _FONT_CACHE[cache_key]
            else:
                return _FONT_CACHE[cache_key]
    
    font_path = None
    font_name = None
    
    # 1) Try language-specific fonts FIRST (bundled fonts)
    if language in _LANGUAGE_FONTS:
        fonts_to_try = _LANGUAGE_FONTS[language]
        if os.path.isdir(_FONTS_DIR):
            for fname in fonts_to_try:
                full = os.path.join(_FONTS_DIR, fname)
                if os.path.isfile(full):
                    font_path = full
                    font_name = fname
                    break
    
    # 2) If not found, search fonts directory with language-specific keywords
    if font_path is None and os.path.isdir(_FONTS_DIR):
        search_keywords = {
            "Urdu": ["nastaliq", "urdu", "jameel", "nafees"],
            "Arabic": ["arabic", "naskh", "amiri", "scheherazade", "lateef", "harmattan"],
            "Hebrew": ["hebrew", "ezra", "sbl", "noto sans hebrew"],
            "Persian": ["persian", "farsi", "vazir", "vazirmatn"],
        }
        
        keywords = search_keywords.get(language, [])
        
        for fname in os.listdir(_FONTS_DIR):
            if not fname.lower().endswith((".ttf", ".otf")):
                continue
            
            fname_lower = fname.lower()
            if any(kw in fname_lower for kw in keywords):
                font_path = os.path.join(_FONTS_DIR, fname)
                font_name = fname
                break
    
    # 3) If still not found, try to download
    if font_path is None:
        font_path = _download_font(language)
        if font_path:
            font_name = os.path.basename(font_path)
    
    # 4) If still not found, try ANY font
    if font_path is None and os.path.isdir(_FONTS_DIR):
        for fname in os.listdir(_FONTS_DIR):
            if fname.lower().endswith((".ttf", ".otf")):
                font_path = os.path.join(_FONTS_DIR, fname)
                font_name = fname
                break
    
    # Cache the result
    _FONT_CACHE[cache_key] = (font_path, font_name)
    if font_path:
        _RTL_FONT_PATH = font_path
        _register_font_with_matplotlib(font_path)
    
    return font_path, font_name


def _register_font_with_matplotlib(font_path: str) -> bool:
    """Register a font with matplotlib."""
    try:
        import matplotlib as mpl
        from matplotlib import font_manager
        
        font_manager.fontManager.addfont(font_path)
        prop = font_manager.FontProperties(fname=font_path)
        font_name = prop.get_name()
        
        mpl.rcParams['font.family'] = font_name
        mpl.rcParams['axes.unicode_minus'] = False
        mpl.rcParams['text.color'] = 'black'
        mpl.rcParams['axes.labelcolor'] = 'black'
        mpl.rcParams['xtick.color'] = 'black'
        mpl.rcParams['ytick.color'] = 'black'
        
        return True
    except Exception as e:
        return False


def set_rtl_font_path(font_path: str) -> bool:
    """Set a custom RTL font path."""
    global _RTL_FONT_PATH, _FONT_REGISTERED
    
    if not os.path.isfile(font_path):
        return False
    
    _RTL_FONT_PATH = font_path
    
    try:
        import matplotlib as mpl
        from matplotlib import font_manager
        
        font_manager.fontManager.addfont(font_path)
        prop = font_manager.FontProperties(fname=font_path)
        font_name = prop.get_name()
        
        mpl.rcParams['font.family'] = font_name
        mpl.rcParams['axes.unicode_minus'] = False
        mpl.rcParams['text.color'] = 'black'
        mpl.rcParams['axes.labelcolor'] = 'black'
        mpl.rcParams['xtick.color'] = 'black'
        mpl.rcParams['ytick.color'] = 'black'
        
        _FONT_REGISTERED = True
        return True
    except Exception as e:
        return False


def get_rtl_font_path(language: str = None) -> str | None:
    """Get the RTL font path for a specific language."""
    font_path, _ = _find_rtl_font(language)
    return font_path


# ============================================================================
# RTL TEXT DETECTION
# ============================================================================

HEBREW_RANGE = (0x0590, 0x05FF)
ARABIC_RANGES = [
    (0x0600, 0x06FF),
    (0x0750, 0x077F),
    (0x08A0, 0x08FF),
    (0xFB50, 0xFDFF),
    (0xFE70, 0xFEFF),
]

def is_rtl_text(text: str) -> bool:
    if not isinstance(text, str):
        return False
    for ch in text:
        cp = ord(ch)
        if HEBREW_RANGE[0] <= cp <= HEBREW_RANGE[1]:
            return True
        for start, end in ARABIC_RANGES:
            if start <= cp <= end:
                return True
    return False


def detect_language(text: str) -> str:
    if not is_rtl_text(text):
        return "unknown"
    for ch in text:
        cp = ord(ch)
        if HEBREW_RANGE[0] <= cp <= HEBREW_RANGE[1]:
            return "Hebrew"
    return "Arabic"


def is_hebrew(text: str) -> bool:
    if not isinstance(text, str):
        return False
    for ch in text:
        if HEBREW_RANGE[0] <= ord(ch) <= HEBREW_RANGE[1]:
            return True
    return False


# ============================================================================
# RENDER FUNCTIONS - COMPLETELY FIXED FOR ALL LANGUAGES
# ============================================================================

def render_rtl_label(
    text: str,
    font_path: str,
    font_size_pt: float = 13,
    dpi: int = 150,
    color: str = "black"
) -> Image.Image:
    """Render RTL text using HarfBuzz + FreeType. Falls back to PIL if needed."""
    if not text or not isinstance(text, str):
        return Image.new("RGBA", (1, 1), (0, 0, 0, 0))
    
    try:
        import uharfbuzz as hb
        import freetype as ft
        
        px_per_pt = dpi / 72.0
        px_size = font_size_pt * px_per_pt
        ft_size = int(px_size * 64)
        
        blob = hb.Blob.from_file_path(font_path)
        face_hb = hb.Face(blob)
        hb_font = hb.Font(face_hb)
        hb_font.scale = (int(px_size * 64), int(px_size * 64))
        
        buf = hb.Buffer()
        buf.add_str(text)
        buf.guess_segment_properties()
        hb.shape(hb_font, buf)
        
        infos = buf.glyph_infos
        positions = buf.glyph_positions
        
        face_ft = ft.Face(font_path)
        face_ft.set_char_size(ft_size)
        
        total_adv = sum(abs(p.x_advance) for p in positions) // 64 + 10
        h = int(px_size * 3.2)
        w = max(int(total_adv) + 10, 10)
        
        canvas = np.zeros((h, w, 4), dtype=np.uint8)
        baseline = int(h * 0.6)
        x = 4
        
        import matplotlib.colors as mcolors
        try:
            rgb = mcolors.to_rgb(color)
            text_rgb = (int(rgb[0] * 255), int(rgb[1] * 255), int(rgb[2] * 255))
        except:
            text_rgb = (0, 0, 0)
        
        for info, pos in zip(infos, positions):
            gid = info.codepoint
            try:
                face_ft.load_glyph(gid, ft.FT_LOAD_RENDER)
            except Exception:
                x += abs(pos.x_advance) // 64
                continue
            
            bm = face_ft.glyph.bitmap
            if bm.width > 0 and bm.rows > 0:
                arr = np.frombuffer(bytes(bm.buffer), dtype=np.uint8).reshape(bm.rows, bm.width)
                bx = x + pos.x_offset // 64 + face_ft.glyph.bitmap_left
                by = baseline - face_ft.glyph.bitmap_top - pos.y_offset // 64
                x1, y1 = max(bx, 0), max(by, 0)
                x2, y2 = min(bx + bm.width, w), min(by + bm.rows, h)
                ax1 = x1 - bx
                ay1 = y1 - by
                ax2 = ax1 + (x2 - x1)
                ay2 = ay1 + (y2 - y1)
                if ax2 > ax1 and ay2 > ay1:
                    for i in range(ay1, ay2):
                        for j in range(ax1, ax2):
                            if arr[i, j] > 0:
                                alpha_val = arr[i, j]
                                canvas[y1 + (i - ay1), x1 + (j - ax1), 0] = text_rgb[0]
                                canvas[y1 + (i - ay1), x1 + (j - ax1), 1] = text_rgb[1]
                                canvas[y1 + (i - ay1), x1 + (j - ax1), 2] = text_rgb[2]
                                canvas[y1 + (i - ay1), x1 + (j - ax1), 3] = alpha_val
            
            x += abs(pos.x_advance) // 64
        
        alpha = canvas[:, :, 3]
        rows = np.any(alpha > 0, axis=1)
        cols = np.any(alpha > 0, axis=0)
        if rows.any() and cols.any():
            r0, r1 = np.where(rows)[0][[0, -1]]
            c0, c1 = np.where(cols)[0][[0, -1]]
            pad = 3
            canvas = canvas[max(r0 - pad, 0):r1 + pad + 1,
                           max(c0 - pad, 0):c1 + pad + 1]
        
        return Image.fromarray(canvas, "RGBA")
        
    except (ImportError, Exception):
        try:
            from PIL import ImageDraw, ImageFont
            img = Image.new("RGBA", (200, 50), (0, 0, 0, 0))
            draw = ImageDraw.Draw(img)
            try:
                font = ImageFont.truetype(font_path, int(font_size_pt))
                draw.text((5, 5), text, font=font, fill=(0, 0, 0, 255))
                bbox = img.getbbox()
                if bbox:
                    img = img.crop(bbox)
                return img
            except Exception:
                pass
        except Exception:
            pass
        
        return Image.new("RGBA", (1, 1), (0, 0, 0, 0))


def render_mixed_label(
    text: str,
    font_path: str,
    font_size_pt: float = 13,
    dpi: int = 150,
    color: str = "black"
) -> Image.Image:
    """
    Render a mixed label that may contain both LTR and RTL text.
    CRITICAL FIX: Only render ONCE, no duplicates.
    """
    if not text or not isinstance(text, str):
        return Image.new("RGBA", (1, 1), (0, 0, 0, 0))
    
    # Check if text contains RTL characters
    has_rtl = is_rtl_text(text)
    
    # If no RTL, render as LTR with PIL
    if not has_rtl:
        return render_ltr_text(text, font_size_pt, dpi, color)
    
    # Check if it's a mixed label with '=' separator
    if '=' in text:
        parts = text.split('=')
        if len(parts) == 2:
            ltr_part = parts[0].strip()
            rtl_part = parts[1].strip()
            
            # Only render RTL part if it actually contains RTL text
            if is_rtl_text(rtl_part):
                # Render LTR prefix
                ltr_img = render_ltr_text(ltr_part + " = ", font_size_pt, dpi, color)
                # Render RTL part
                rtl_img = render_rtl_label(rtl_part, font_path, font_size_pt, dpi, color)
                
                # Combine images
                total_width = ltr_img.width + rtl_img.width + 2
                max_height = max(ltr_img.height, rtl_img.height)
                combined = Image.new("RGBA", (total_width, max_height), (0, 0, 0, 0))
                combined.paste(ltr_img, (0, (max_height - ltr_img.height) // 2))
                combined.paste(rtl_img, (ltr_img.width + 2, (max_height - rtl_img.height) // 2))
                return combined
            else:
                # RTL part is not actually RTL, render entire thing as LTR
                return render_ltr_text(text, font_size_pt, dpi, color)
    
    # Pure RTL text
    return render_rtl_label(text, font_path, font_size_pt, dpi, color)


def render_ltr_text(
    text: str,
    font_size_pt: float = 13,
    dpi: int = 150,
    color: str = "black"
) -> Image.Image:
    """Render LTR text using PIL."""
    try:
        from PIL import ImageDraw, ImageFont
        import matplotlib.colors as mcolors
        
        # Try to use a standard font
        font = None
        for font_name in ["Arial", "Helvetica", "DejaVuSans", "sans-serif"]:
            try:
                font = ImageFont.truetype(font_name, int(font_size_pt))
                break
            except:
                continue
        
        if font is None:
            font = ImageFont.load_default()
        
        # Calculate text size
        temp_img = Image.new("RGBA", (1, 1), (0, 0, 0, 0))
        temp_draw = ImageDraw.Draw(temp_img)
        bbox = temp_draw.textbbox((0, 0), text, font=font)
        width = bbox[2] - bbox[0] + 10
        height = bbox[3] - bbox[1] + 10
        
        # Create image
        img = Image.new("RGBA", (width, height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        
        # Convert color
        try:
            rgb = mcolors.to_rgb(color)
            text_color = (int(rgb[0] * 255), int(rgb[1] * 255), int(rgb[2] * 255), 255)
        except:
            text_color = (0, 0, 0, 255)
        
        # Draw text
        draw.text((2, 2), text, font=font, fill=text_color)
        
        # Crop to text
        bbox = img.getbbox()
        if bbox:
            img = img.crop(bbox)
        
        return img
    except Exception as e:
        return Image.new("RGBA", (1, 1), (0, 0, 0, 0))


# ============================================================================
# RTL PLOT UTILITIES - FIXED TO AVOID DUPLICATES
# ============================================================================

def apply_rtl_ytick_images(
    ax,
    tick_labels,
    font_path,
    font_size_pt=13,
    dpi=150,
    ha="right",
    color="black",
    x_offset_pts=-6
):
    """Replace y-axis text tick labels with RTL-rendered images."""
    import matplotlib.pyplot as plt
    from matplotlib.offsetbox import AnnotationBbox, OffsetImage
    
    if not tick_labels or not font_path:
        return
    
    fig = ax.figure
    fig.canvas.draw()
    
    yticks = ax.get_yticks()
    ylim = ax.get_ylim()
    ylim_lo = min(ylim) - 0.5
    ylim_hi = max(ylim) + 0.5
    
    if isinstance(tick_labels, np.ndarray):
        tick_labels = tick_labels.tolist()
    
    # Track which positions already have rendered labels to avoid duplicates
    rendered_positions = set()
    
    for ytick_pos, label_text in zip(yticks, tick_labels):
        # Skip if not RTL text
        if not is_rtl_text(str(label_text)):
            continue
        
        # Skip if out of bounds
        if not (ylim_lo <= ytick_pos <= ylim_hi):
            continue
        
        # CRITICAL FIX: Skip if we already rendered at this position
        pos_rounded = round(ytick_pos, 6)
        if pos_rounded in rendered_positions:
            continue
        
        # Determine if mixed label
        if '=' in str(label_text):
            img_pil = render_mixed_label(
                str(label_text),
                font_path,
                font_size_pt=font_size_pt,
                dpi=dpi,
                color=color
            )
        else:
            img_pil = render_rtl_label(
                str(label_text),
                font_path,
                font_size_pt=font_size_pt,
                dpi=dpi,
                color=color
            )
        
        if img_pil.size == (1, 1):
            continue
        
        arr = np.array(img_pil, dtype=np.uint8).copy()
        
        # Apply color if not black
        if color != "black" and color != "#000000":
            import matplotlib.colors as mcolors
            try:
                rgb = mcolors.to_rgb(color)
                for i in range(arr.shape[0]):
                    for j in range(arr.shape[1]):
                        if arr[i, j, 3] > 0:
                            arr[i, j, 0] = int(rgb[0] * 255)
                            arr[i, j, 1] = int(rgb[1] * 255)
                            arr[i, j, 2] = int(rgb[2] * 255)
            except Exception:
                pass
        
        oi = OffsetImage(arr, zoom=72.0 / dpi)
        oi.image.axes = ax
        
        ab = AnnotationBbox(
            oi,
            xy=(0, ytick_pos),
            xycoords=("axes fraction", "data"),
            xybox=(x_offset_pts, 0),
            boxcoords="offset points",
            frameon=False,
            box_alignment=(1.0, 0.5),
            pad=0,
        )
        ax.add_artist(ab)
        
        # Mark this position as rendered
        rendered_positions.add(pos_rounded)


def hide_rtl_text_ticks(ax, tick_labels):
    """Make the original text y-tick labels invisible for any RTL entries."""
    if not tick_labels:
        return
    
    text_ticks = ax.get_yticklabels()
    for txt, label_text in zip(text_ticks, tick_labels):
        if is_rtl_text(str(label_text)):
            txt.set_visible(False)


def _adjust_left_margin_for_rtl_labels(
    ax,
    tick_labels,
    font_path,
    font_size_pt=12,
    dpi=150,
    extra_pad_in=0.20
):
    """Widen the figure left margin so RTL y-tick images don't get cropped."""
    if not tick_labels:
        return
    
    rtl_labels = [str(lbl) for lbl in tick_labels if is_rtl_text(str(lbl))]
    if not rtl_labels:
        return
    
    max_px = 0
    for lbl in rtl_labels:
        try:
            img = render_rtl_label(lbl, font_path, font_size_pt=font_size_pt, dpi=dpi)
            max_px = max(max_px, img.size[0])
        except:
            continue
    
    if max_px < 2:
        return
    
    fig = ax.figure
    fig_dpi = fig.dpi
    fig_w_in = fig.get_size_inches()[0]
    label_w_in = (max_px * 72.0 / dpi) / fig_dpi
    needed_left = (label_w_in + extra_pad_in) / fig_w_in
    
    if needed_left > fig.subplotpars.left:
        fig.subplots_adjust(left=min(needed_left, 0.55))


def add_rtl_title(
    text: str,
    ax=None,
    font_size_pt: float = 14,
    dpi: int = 150,
    pad: float = 15,
    color: str = "black"
):
    """Place a HarfBuzz-rendered RTL title centered above the axes."""
    import matplotlib.pyplot as plt
    from matplotlib.offsetbox import AnnotationBbox, OffsetImage
    
    if ax is None:
        ax = plt.gca()
    
    if not text:
        return
    
    font_path, _ = _find_rtl_font()
    if font_path is None:
        ax.set_title(text, fontsize=font_size_pt, color=color)
        return
    
    img_pil = render_rtl_label(
        text,
        font_path,
        font_size_pt=font_size_pt,
        dpi=dpi,
        color=color
    )
    if img_pil.width < 2:
        ax.set_title(text, fontsize=font_size_pt, color=color)
        return
    
    arr = np.array(img_pil, dtype=np.uint8)
    oi = OffsetImage(arr, zoom=72.0 / dpi)
    oi.image.axes = ax
    
    ab = AnnotationBbox(
        oi,
        xy=(0.5, 1.0),
        xycoords="axes fraction",
        xybox=(0, pad),
        boxcoords="offset points",
        frameon=False,
        box_alignment=(0.5, 0.0),
        clip_on=False,
        pad=0,
    )
    ax.add_artist(ab)


# ============================================================================
# AUTO-INITIALIZATION
# ============================================================================

def _auto_initialize():
    """Auto-initialize RTL font support."""
    try:
        import matplotlib as mpl
        mpl.rcParams['text.color'] = 'black'
        mpl.rcParams['axes.labelcolor'] = 'black'
        mpl.rcParams['xtick.color'] = 'black'
        mpl.rcParams['ytick.color'] = 'black'
    except:
        pass
    
    print("[INFO] RTL font support ready.")
    print("[INFO] Supported languages: Arabic, Urdu, Persian, Hebrew")


# Run auto-initialization
_auto_initialize()


# ============================================================================
# EXPORTED FUNCTIONS
# ============================================================================

__all__ = [
    '_find_rtl_font',
    'set_rtl_font_path',
    'get_rtl_font_path',
    'set_current_language',
    'get_current_language',
    'is_rtl_text',
    'detect_language',
    'is_hebrew',
    'render_rtl_label',
    'render_mixed_label',
    'render_ltr_text',
    'apply_rtl_ytick_images',
    'hide_rtl_text_ticks',
    '_adjust_left_margin_for_rtl_labels',
    'add_rtl_title',
    '_LANGUAGE_FONTS',
]