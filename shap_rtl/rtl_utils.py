"""
RTL (Right-to-Left) text utilities for SHAP_RTL_package.

Supports ALL RTL languages including:
- Arabic (العربية)
- Urdu (اردو)
- Persian/Farsi (فارسی)
- Hebrew (עברית)
- Pashto (پښتو)
- Dari (دری)
- Kurdish (کوردی)
- Sindhi (سنڌي)
- Balochi (بلوچی)
- Kashmiri (कॉशुर / کٲشُر)
- Punjabi (Shahmukhi - پنجابی)

Strategy
--------
Matplotlib does NOT drive an OpenType shaping engine, so passing Arabic
presentation-form codepoints (U+FBxx / U+FExx) through FontProperties
produces tofu boxes even when a correct Nastaliq font is registered.

The fix used here: render every RTL label out-of-band with
  HarfBuzz  → proper glyph shaping
  FreeType  → pixel-level rasterisation
  PIL/Pillow → RGBA image

then splice that image into the matplotlib Axes as an AnnotationBbox /
OffsetImage instead of a text tick.  All original tick *positions* are kept
exactly; we just hide the text tick and overlay the image at the same spot.
"""

from __future__ import annotations

import os
import warnings
import numpy as np
from PIL import Image

# ── font location ──────────────────────────────────────────────────────────────
_PKG_DIR = os.path.dirname(os.path.abspath(__file__))
_FONTS_DIR = os.path.join(os.path.dirname(_PKG_DIR), "fonts")

# ============================================================================
# FONT CONFIGURATION - Supports ALL RTL Languages
# ============================================================================

# Language-specific font preferences
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
        "Arial.ttf",  # Arial has Hebrew support
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
        "NotoSansHebrew-Regular.ttf",  # For Kurdish in Hebrew script
    ],
    "Sindhi": [
        "NotoSansArabic-Regular.ttf",
        "NotoNaskhArabic-Regular.ttf",
        "Amiri-Regular.ttf",
        "JameelNooriNastaliq.ttf",
    ],
    "Balochi": [
        "NotoSansArabic-Regular.ttf",
        "Amiri-Regular.ttf",
        "ScheherazadeNew-Regular.ttf",
    ],
    "Kashmiri": [
        "NotoSansArabic-Regular.ttf",
        "Amiri-Regular.ttf",
        "NotoSansDevanagari-Regular.ttf",  # For Kashmiri in Devanagari
    ],
    "Punjabi_Shahmukhi": [
        "NotoSansArabic-Regular.ttf",
        "NotoNastaliqUrdu-Regular.ttf",
        "JameelNooriNastaliq.ttf",
    ],
}

# Fallback fonts (tried in order)
_FALLBACK_FONTS = [
    "NotoSansArabic-Regular.ttf",
    "NotoNaskhArabic-Regular.ttf",
    "NotoSansHebrew-Regular.ttf",
    "NotoNastaliqUrdu-Regular.ttf",
    "Amiri-Regular.ttf",
    "ScheherazadeNew-Regular.ttf",
    "Arial.ttf",
    "TimesNewRoman.ttf",
]

# Font names for matplotlib
_PREFERRED_FONT_NAMES = [
    "Jameel Noori Nastaliq",
    "Noto Nastaliq Urdu",
    "Noto Sans Arabic",
    "Noto Sans Hebrew",
    "Amiri",
    "Scheherazade New",
    "Arial",
    "Times New Roman",
]

# ============================================================================
# RTL LANGUAGE CONFIGURATION
# ============================================================================

# Supported RTL languages and their Unicode ranges
_RTL_LANGUAGES = {
    "Arabic": {
        "name": "Arabic",
        "code": "ar",
        "unicode_ranges": [(0x0600, 0x06FF), (0x0750, 0x077F), (0x08A0, 0x08FF), (0xFB50, 0xFDFF), (0xFE70, 0xFEFF)],
        "sample": "السلام عليكم",
    },
    "Urdu": {
        "name": "Urdu",
        "code": "ur",
        "unicode_ranges": [(0x0600, 0x06FF), (0x0750, 0x077F), (0x08A0, 0x08FF), (0xFB50, 0xFDFF), (0xFE70, 0xFEFF)],
        "sample": "السلام علیکم",
    },
    "Persian": {
        "name": "Persian",
        "code": "fa",
        "unicode_ranges": [(0x0600, 0x06FF), (0x0750, 0x077F), (0x08A0, 0x08FF), (0xFB50, 0xFDFF), (0xFE70, 0xFEFF)],
        "sample": "سلام",
    },
    "Hebrew": {
        "name": "Hebrew",
        "code": "he",
        "unicode_ranges": [(0x0590, 0x05FF)],
        "sample": "שלום",
    },
    "Pashto": {
        "name": "Pashto",
        "code": "ps",
        "unicode_ranges": [(0x0600, 0x06FF), (0x0750, 0x077F), (0x08A0, 0x08FF), (0xFB50, 0xFDFF), (0xFE70, 0xFEFF)],
        "sample": "سلام",
    },
    "Dari": {
        "name": "Dari",
        "code": "prs",
        "unicode_ranges": [(0x0600, 0x06FF), (0x0750, 0x077F), (0x08A0, 0x08FF), (0xFB50, 0xFDFF), (0xFE70, 0xFEFF)],
        "sample": "سلام",
    },
    "Kurdish": {
        "name": "Kurdish",
        "code": "ku",
        "unicode_ranges": [(0x0600, 0x06FF), (0x0750, 0x077F), (0x08A0, 0x08FF), (0xFB50, 0xFDFF), (0xFE70, 0xFEFF)],
        "sample": "سڵاو",
    },
    "Sindhi": {
        "name": "Sindhi",
        "code": "sd",
        "unicode_ranges": [(0x0600, 0x06FF), (0x0750, 0x077F), (0x08A0, 0x08FF), (0xFB50, 0xFDFF), (0xFE70, 0xFEFF)],
        "sample": "سلام",
    },
}

# Combined RTL Unicode ranges for detection
_RTL_RANGES = []
for lang_info in _RTL_LANGUAGES.values():
    _RTL_RANGES.extend(lang_info["unicode_ranges"])
# Remove duplicates
_RTL_RANGES = list(set(_RTL_RANGES))

_RTL_FONT_PATH: str | None = None
_LATIN_FONT_PATH: str | None = None
_FONT_REGISTERED: bool = False
_CURRENT_LANGUAGE: str = "auto"

# ============================================================================
# FONT MANAGEMENT FUNCTIONS
# ============================================================================

def _find_latin_font() -> str | None:
    """Return a path to a Latin/system font suitable for rendering LTR text."""
    global _LATIN_FONT_PATH
    if _LATIN_FONT_PATH is not None:
        return _LATIN_FONT_PATH
    
    try:
        import matplotlib.font_manager as fm
        for font_name in ["DejaVu Sans", "Arial", "Helvetica", "sans-serif"]:
            try:
                path = fm.findfont(fm.FontProperties(family=font_name))
                if path and os.path.isfile(path):
                    _LATIN_FONT_PATH = path
                    return _LATIN_FONT_PATH
            except Exception:
                continue
    except Exception:
        pass
    
    # Fallback: common system paths
    for path in [
        "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/Arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/Library/Fonts/Arial.ttf",
        "/System/Library/Fonts/Helvetica.ttf",
    ]:
        if os.path.isfile(path):
            _LATIN_FONT_PATH = path
            return _LATIN_FONT_PATH
    
    return None


def _get_fonts_for_language(language: str) -> list:
    """Get list of font files to try for a specific language."""
    fonts = []
    
    # Get language-specific fonts
    if language in _LANGUAGE_FONTS:
        fonts.extend(_LANGUAGE_FONTS[language])
    
    # Add fallback fonts
    fonts.extend(_FALLBACK_FONTS)
    
    # Remove duplicates while preserving order
    seen = set()
    return [f for f in fonts if not (f in seen or seen.add(f))]


def _register_font_with_matplotlib(font_path: str) -> bool:
    """
    Register a font with matplotlib's font manager.
    This is critical for preventing glyph missing errors.
    """
    global _FONT_REGISTERED
    
    if _FONT_REGISTERED:
        return True
    
    try:
        import matplotlib.font_manager as fm
        import matplotlib.pyplot as plt
        
        # Add the font to matplotlib's font manager
        fm.fontManager.addfont(font_path)
        
        # Get the font name
        prop = fm.FontProperties(fname=font_path)
        font_name = prop.get_name()
        
        # Set font family hierarchy with fallback for English
        # This prevents glyph missing errors by using RTL font for RTL text
        # and fallback to Latin fonts for English text
        plt.rcParams['font.family'] = [
            font_name,           # Primary: RTL font
            'DejaVu Sans',       # Fallback 1: Good for English
            'Arial',             # Fallback 2: Common system font
            'sans-serif'         # Final fallback
        ]
        plt.rcParams['axes.unicode_minus'] = False
        
        # Clear font cache to ensure new font is used
        try:
            fm.fontManager.ttflist = fm.fontManager.ttflist
        except Exception:
            pass
        
        _FONT_REGISTERED = True
        return True
        
    except Exception as e:
        print(f"[WARNING] Could not register font with matplotlib: {e}")
        return False


def _find_rtl_font(language: str = None) -> tuple[str | None, str]:
    """
    Return (path, display_name) for the best available RTL font.
    
    Parameters
    ----------
    language : str, optional
        Language name to find specific font for.
        If None, tries to find any RTL font.
    """
    global _RTL_FONT_PATH, _FONT_REGISTERED, _CURRENT_LANGUAGE

    if language:
        _CURRENT_LANGUAGE = language

    if _RTL_FONT_PATH is not None:
        return _RTL_FONT_PATH, os.path.splitext(os.path.basename(_RTL_FONT_PATH))[0]

    # 1) Try language-specific fonts in bundled fonts dir
    if os.path.isdir(_FONTS_DIR):
        # Get fonts for this language
        fonts_to_try = _get_fonts_for_language(language) if language else _FALLBACK_FONTS
        
        for fname in fonts_to_try:
            full = os.path.join(_FONTS_DIR, fname)
            if os.path.isfile(full):
                _RTL_FONT_PATH = full
                _register_font_with_matplotlib(full)
                return _RTL_FONT_PATH, fname

        # any .ttf/.otf in the dir
        for fname in os.listdir(_FONTS_DIR):
            if fname.lower().endswith((".ttf", ".otf")):
                _RTL_FONT_PATH = os.path.join(_FONTS_DIR, fname)
                _register_font_with_matplotlib(_RTL_FONT_PATH)
                return _RTL_FONT_PATH, fname

    # 2) system font search
    try:
        import matplotlib.font_manager as fm
        all_fonts = {f.name.lower(): f.fname for f in fm.fontManager.ttflist}
        
        # Try language-specific font names
        if language and language in _LANGUAGE_FONTS:
            for font_file in _LANGUAGE_FONTS[language]:
                font_name = os.path.splitext(font_file)[0].lower()
                if font_name in all_fonts:
                    _RTL_FONT_PATH = all_fonts[font_name]
                    _register_font_with_matplotlib(_RTL_FONT_PATH)
                    return _RTL_FONT_PATH, font_name
        
        # Try preferred font names
        for name in _PREFERRED_FONT_NAMES:
            if name.lower() in all_fonts:
                _RTL_FONT_PATH = all_fonts[name.lower()]
                _register_font_with_matplotlib(_RTL_FONT_PATH)
                return _RTL_FONT_PATH, name
    except Exception:
        pass

    # 3) Try to find any RTL font in system
    try:
        import matplotlib.font_manager as fm
        rtl_keywords = ['arabic', 'urdu', 'nastaliq', 'hebrew', 'persian', 'pashto', 'sindhi']
        for font in fm.fontManager.ttflist:
            font_lower = font.name.lower()
            if any(keyword in font_lower for keyword in rtl_keywords):
                _RTL_FONT_PATH = font.fname
                _register_font_with_matplotlib(_RTL_FONT_PATH)
                return _RTL_FONT_PATH, font.name
    except Exception:
        pass

    # 4) Final fallback: Use system default
    try:
        import matplotlib.pyplot as plt
        plt.rcParams['font.family'] = ['DejaVu Sans', 'Arial', 'sans-serif']
        plt.rcParams['axes.unicode_minus'] = False
    except Exception:
        pass

    warnings.warn(
        f"No RTL font found for language '{language}'. "
        f"Place font files in the 'fonts/' directory of the package. "
        f"Using system fallback.",
        UserWarning, stacklevel=3,
    )
    return None, "default"


def get_rtl_font_path(language: str = None) -> str | None:
    """Get the RTL font path (public API)."""
    font_path, _ = _find_rtl_font(language)
    return font_path


def set_rtl_font_path(font_path: str) -> bool:
    """Set a custom RTL font path."""
    global _RTL_FONT_PATH
    if not os.path.isfile(font_path):
        print(f"[ERROR] Font not found: {font_path}")
        return False
    
    _RTL_FONT_PATH = font_path
    return _register_font_with_matplotlib(font_path)


# ============================================================================
# AUTO-INITIALIZATION
# ============================================================================

def _auto_initialize():
    """Auto-initialize RTL font support on import."""
    try:
        font_path, font_name = _find_rtl_font()
        if font_path:
            print(f"[INFO] RTL font initialized: {font_name}")
        else:
            print("[INFO] No RTL font found. RTL text may not render correctly.")
            print("[INFO] Supported fonts for each language:")
            for lang in _RTL_LANGUAGES.keys():
                fonts = _LANGUAGE_FONTS.get(lang, [])
                if fonts:
                    print(f"  - {lang}: {fonts[0]}")
    except Exception as e:
        print(f"[WARNING] Auto-initialization failed: {e}")


# Run auto-initialization
_auto_initialize()


# ============================================================================
# RTL TEXT DETECTION AND PROCESSING
# ============================================================================

def is_rtl_text(text: str) -> bool:
    """
    True if the string contains any RTL codepoints.
    Supports all RTL languages.
    """
    if not isinstance(text, str):
        return False
    for ch in text:
        cp = ord(ch)
        for start, end in _RTL_RANGES:
            if start <= cp <= end:
                return True
    return False


def detect_language(text: str) -> str:
    """
    Detect the likely RTL language of the text.
    
    Returns
    -------
    str
        Language name or "unknown"
    """
    if not is_rtl_text(text):
        return "unknown"
    
    # Hebrew has a distinct range
    hebrew_range = (0x0590, 0x05FF)
    for ch in text:
        cp = ord(ch)
        if hebrew_range[0] <= cp <= hebrew_range[1]:
            return "Hebrew"
    
    # Check for Arabic script languages (most use Arabic script)
    # This is a simplified detection
    return "Arabic"  # Default to Arabic for Arabic script


def process_label(text: str, force_rtl: bool = False) -> str:
    """Return the label unchanged – reshaping is handled at render time."""
    return str(text) if not isinstance(text, str) else text


def reshape_rtl(text: str) -> str:
    """
    Return a reshaped/display-ordered string for RTL languages.

    Uses `arabic_reshaper` + `python-bidi` if available; otherwise returns
    the original text unchanged.
    """
    if not isinstance(text, str):
        return text
    try:
        import arabic_reshaper
        from bidi.algorithm import get_display
        reshaped = arabic_reshaper.reshape(text)
        return get_display(reshaped)
    except Exception:
        return text


def process_labels(labels, force_rtl: bool = False):
    return [process_label(str(lbl)) for lbl in labels]


# ============================================================================
# HARFBUZZ + FREETYPE RENDERER
# ============================================================================

_HB_OK = False
_FT_OK = False

try:
    import uharfbuzz as hb
    _HB_OK = True
except ImportError:
    pass

try:
    import freetype as ft
    _FT_OK = True
except ImportError:
    pass


def render_rtl_label(
    text: str,
    font_path: str,
    font_size_pt: float = 13,
    dpi: int = 150,
) -> Image.Image:
    """
    Render *text* using HarfBuzz shaping + FreeType rasterisation.

    Returns an RGBA PIL Image with transparent background and black ink.
    Falls back to a blank 1×1 image if shaping libraries are unavailable.
    """
    if not (_HB_OK and _FT_OK):
        if not _HB_OK:
            warnings.warn("uharfbuzz not installed – RTL labels will be plain text. "
                          "Run: pip install uharfbuzz", UserWarning, stacklevel=2)
        if not _FT_OK:
            warnings.warn("freetype-py not installed – RTL labels will be plain text. "
                          "Run: pip install freetype-py", UserWarning, stacklevel=2)
        return Image.new("RGBA", (1, 1), (0, 0, 0, 0))

    import uharfbuzz as hb
    import freetype as ft

    px_per_pt = dpi / 72.0
    px_size   = font_size_pt * px_per_pt
    ft_size   = int(px_size * 64)

    # ── HarfBuzz shaping ──────────────────────────────────────────────────────
    blob    = hb.Blob.from_file_path(font_path)
    face_hb = hb.Face(blob)
    hb_font = hb.Font(face_hb)
    hb_font.scale = (int(px_size * 64), int(px_size * 64))

    buf = hb.Buffer()
    buf.add_str(text)
    buf.guess_segment_properties()
    hb.shape(hb_font, buf)

    infos     = buf.glyph_infos
    positions = buf.glyph_positions

    # ── FreeType rasterisation ────────────────────────────────────────────────
    face_ft = ft.Face(font_path)
    face_ft.set_char_size(ft_size)

    total_adv = sum(abs(p.x_advance) for p in positions) // 64 + 10
    h = int(px_size * 3.2)
    w = max(int(total_adv) + 10, 10)

    canvas   = np.zeros((h, w, 4), dtype=np.uint8)
    baseline = int(h * 0.6)
    x = 4

    for info, pos in zip(infos, positions):
        gid = info.codepoint
        try:
            face_ft.load_glyph(gid, ft.FT_LOAD_RENDER)
        except Exception:
            x += abs(pos.x_advance) // 64
            continue

        bm = face_ft.glyph.bitmap
        if bm.width > 0 and bm.rows > 0:
            arr  = np.frombuffer(bytes(bm.buffer), dtype=np.uint8).reshape(bm.rows, bm.width)
            bx   = x + pos.x_offset // 64 + face_ft.glyph.bitmap_left
            by   = baseline - face_ft.glyph.bitmap_top - pos.y_offset // 64
            x1, y1 = max(bx, 0), max(by, 0)
            x2, y2 = min(bx + bm.width, w), min(by + bm.rows, h)
            ax1 = x1 - bx;  ay1 = y1 - by
            ax2 = ax1 + (x2 - x1); ay2 = ay1 + (y2 - y1)
            if ax2 > ax1 and ay2 > ay1:
                canvas[y1:y2, x1:x2, 3] = np.maximum(
                    canvas[y1:y2, x1:x2, 3], arr[ay1:ay2, ax1:ax2]
                )

        x += abs(pos.x_advance) // 64

    # trim transparent rows/cols
    alpha = canvas[:, :, 3]
    rows  = np.any(alpha > 0, axis=1)
    cols  = np.any(alpha > 0, axis=0)
    if rows.any() and cols.any():
        r0, r1 = np.where(rows)[0][[0, -1]]
        c0, c1 = np.where(cols)[0][[0, -1]]
        pad = 3
        canvas = canvas[max(r0 - pad, 0):r1 + pad + 1,
                        max(c0 - pad, 0):c1 + pad + 1]

    return Image.fromarray(canvas, "RGBA")


def _render_latin_with_freetype(
    text: str,
    font_size_pt: float = 12,
    dpi: int = 150,
) -> Image.Image:
    """Render plain LTR/ASCII text using FreeType with a Latin system font."""
    if not _FT_OK:
        return Image.new("RGBA", (1, 1), (0, 0, 0, 0))

    latin_font = _find_latin_font()
    if latin_font is None:
        return Image.new("RGBA", (1, 1), (0, 0, 0, 0))

    import freetype as ft

    px_per_pt = dpi / 72.0
    px_size   = font_size_pt * px_per_pt
    ft_size   = int(px_size * 64)

    face = ft.Face(latin_font)
    face.set_char_size(ft_size)

    total_adv = 0
    for ch in text:
        try:
            face.load_char(ch, ft.FT_LOAD_DEFAULT)
            total_adv += face.glyph.advance.x >> 6
        except Exception:
            total_adv += int(px_size * 0.5)

    h = int(px_size * 2.2)
    w = max(int(total_adv) + 10, 10)
    canvas   = np.zeros((h, w, 4), dtype=np.uint8)
    baseline = int(h * 0.68)
    x = 2

    for ch in text:
        try:
            face.load_char(ch, ft.FT_LOAD_RENDER)
            bm = face.glyph.bitmap
            if bm.width > 0 and bm.rows > 0:
                arr  = np.frombuffer(bytes(bm.buffer), dtype=np.uint8).reshape(bm.rows, bm.width)
                bx   = x + face.glyph.bitmap_left
                by   = baseline - face.glyph.bitmap_top
                x1, y1 = max(bx, 0), max(by, 0)
                x2, y2 = min(bx + bm.width, w), min(by + bm.rows, h)
                ax1 = x1 - bx; ay1 = y1 - by
                ax2 = ax1 + (x2 - x1); ay2 = ay1 + (y2 - y1)
                if ax2 > ax1 and ay2 > ay1:
                    canvas[y1:y2, x1:x2, 3] = np.maximum(
                        canvas[y1:y2, x1:x2, 3], arr[ay1:ay2, ax1:ax2]
                    )
            x += face.glyph.advance.x >> 6
        except Exception:
            x += int(px_size * 0.5)

    alpha = canvas[:, :, 3]
    rows  = np.any(alpha > 0, axis=1)
    cols  = np.any(alpha > 0, axis=0)
    if rows.any() and cols.any():
        r0, r1 = np.where(rows)[0][[0, -1]]
        c0, c1 = np.where(cols)[0][[0, -1]]
        pad = 2
        canvas = canvas[max(r0 - pad, 0):r1 + pad + 1,
                        max(c0 - pad, 0):c1 + pad + 1]

    return Image.fromarray(canvas, "RGBA")


def render_mixed_label(
    text: str,
    font_path: str,
    font_size_pt: float = 12,
    dpi: int = 150,
) -> Image.Image:
    """
    Render a label that contains both LTR and RTL parts.
    """
    if " = " in text:
        ltr_part, rtl_part = text.split(" = ", 1)
        if is_rtl_text(rtl_part):
            ltr_img = _render_latin_with_freetype(ltr_part + " = ", font_size_pt, dpi)
            rtl_img = render_rtl_label(rtl_part, font_path, font_size_pt, dpi)

            if ltr_img.width < 2 and rtl_img.width < 2:
                return Image.new("RGBA", (1, 1), (0, 0, 0, 0))

            h = max(ltr_img.height, rtl_img.height, 1)
            w = ltr_img.width + rtl_img.width
            combined = Image.new("RGBA", (max(w, 1), h), (0, 0, 0, 0))
            combined.paste(ltr_img, (0, (h - ltr_img.height) // 2))
            if rtl_img.width >= 2:
                combined.paste(
                    rtl_img,
                    (ltr_img.width, (h - rtl_img.height) // 2),
                    rtl_img,
                )
            return combined

    return render_rtl_label(text, font_path, font_size_pt, dpi)


def add_rtl_title(text: str, ax=None, font_size_pt: float = 14,
                  dpi: int = 150, pad: float = 15):
    """
    Place a HarfBuzz-rendered RTL title centered above the axes.
    """
    import matplotlib.pyplot as plt
    from matplotlib.offsetbox import AnnotationBbox, OffsetImage

    if ax is None:
        ax = plt.gca()

    font_path, _ = _find_rtl_font()
    if font_path is None:
        ax.set_title(text, fontsize=font_size_pt)
        return

    img_pil = render_rtl_label(text, font_path, font_size_pt=font_size_pt, dpi=dpi)
    if img_pil.width < 2:
        ax.set_title(text, fontsize=font_size_pt)
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


def apply_rtl_ytick_images(
    ax,
    tick_labels: list[str],
    font_path: str,
    font_size_pt: float = 13,
    dpi: int = 150,
    ha: str = "right",
    color: str = "black",
    x_offset_pts: float = -6,
):
    """
    Replace y-axis text tick labels that contain RTL characters with
    properly-shaped images rendered via HarfBuzz+FreeType.
    """
    import matplotlib.pyplot as plt
    from matplotlib.offsetbox import AnnotationBbox, OffsetImage

    fig = ax.figure
    fig.canvas.draw()

    yticks = ax.get_yticks()
    ylim   = ax.get_ylim()
    ylim_lo = min(ylim) - 0.5
    ylim_hi = max(ylim) + 0.5

    for ytick_pos, label_text in zip(yticks, tick_labels):
        if not is_rtl_text(label_text):
            continue

        if not (ylim_lo <= ytick_pos <= ylim_hi):
            continue

        img_pil = render_rtl_label(label_text, font_path,
                                   font_size_pt=font_size_pt, dpi=dpi)
        if img_pil.size == (1, 1):
            continue

        arr = np.array(img_pil, dtype=np.uint8).copy()
        if color != "black" and color != "#000000":
            import matplotlib.colors as mcolors
            rgb = mcolors.to_rgb(color)
            arr[:, :, 0] = int(rgb[0] * 255)
            arr[:, :, 1] = int(rgb[1] * 255)
            arr[:, :, 2] = int(rgb[2] * 255)

        oi = OffsetImage(arr, zoom=72.0 / dpi)
        oi.image.axes = ax

        ab = AnnotationBbox(
            oi,
            xy=(0, ytick_pos),
            xycoords=("axes fraction", "data"),
            xybox=(-8, 0),
            boxcoords="offset points",
            frameon=False,
            box_alignment=(1.0, 0.5),
            pad=0,
        )
        ax.add_artist(ab)

    _adjust_left_margin_for_rtl_labels(ax, tick_labels, font_path,
                                       font_size_pt=font_size_pt, dpi=dpi)


def _adjust_left_margin_for_rtl_labels(
    ax,
    tick_labels: list[str],
    font_path: str,
    font_size_pt: float = 12,
    dpi: int = 150,
    extra_pad_in: float = 0.20,
):
    """Widen the figure left margin so RTL y-tick images don't get cropped."""
    rtl_labels = [lbl for lbl in tick_labels if is_rtl_text(lbl)]
    if not rtl_labels:
        return

    max_px = max(
        render_rtl_label(lbl, font_path, font_size_pt=font_size_pt, dpi=dpi).size[0]
        for lbl in rtl_labels
    )
    if max_px < 2:
        return

    fig = ax.figure
    fig_dpi      = fig.dpi
    fig_w_in     = fig.get_size_inches()[0]
    label_w_in   = (max_px * 72.0 / dpi) / fig_dpi
    needed_left  = (label_w_in + extra_pad_in) / fig_w_in

    if needed_left > fig.subplotpars.left:
        fig.subplots_adjust(left=min(needed_left, 0.55))


def hide_rtl_text_ticks(ax, tick_labels: list[str]):
    """
    Make the original text y-tick labels invisible for any RTL entries.
    This prevents the original matplotlib text labels from showing
    behind the rendered RTL images.
    """
    text_ticks = ax.get_yticklabels()
    for txt, label_text in zip(text_ticks, tick_labels):
        if is_rtl_text(label_text):
            txt.set_visible(False)


# ============================================================================
# PUBLIC API - EXPORTED FUNCTIONS
# ============================================================================

__all__ = [
    # Font management
    '_find_rtl_font',
    '_find_latin_font',
    '_register_font_with_matplotlib',
    'get_rtl_font_path',
    'set_rtl_font_path',
    
    # RTL detection
    'is_rtl_text',
    'detect_language',
    'process_label',
    'process_labels',
    'reshape_rtl',
    
    # Rendering
    'render_rtl_label',
    'render_mixed_label',
    '_render_latin_with_freetype',
    
    # Plot utilities
    'apply_rtl_ytick_images',
    'hide_rtl_text_ticks',
    '_adjust_left_margin_for_rtl_labels',
    'add_rtl_title',
    
    # Constants
    '_RTL_LANGUAGES',
    '_RTL_RANGES',
    '_LANGUAGE_FONTS',
]