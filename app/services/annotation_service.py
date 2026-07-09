"""
annotation_service.py
=====================
Professional weld defect annotation using Pillow — v5.

PRIORITY DEFECTS ONLY:
    Undercut          → RED       (255, 50, 50)   — drawn as RECTANGLE
    Underfill         → BLUE      (30, 120, 255)  — drawn as RECTANGLE
    Excess Reinforce  → MAGENTA   (255, 50, 255)  — drawn as OVAL (ellipse)
    Blowhole/Porosity → CYAN      (0, 210, 230)   — drawn as CIRCLE

All other defect types (spatter, crack, overlap, etc.) are filtered out
and not drawn or shown in the legend.
"""

from PIL import Image, ImageDraw, ImageFont
import io
import math
import random
import textwrap
from typing import List, Tuple, Optional
from app.schemas.inspection import Defect
from loguru import logger

# ── Priority-only color palette ───────────────────────────────────────────────
DEFECT_COLORS: dict = {
    "blowhole":             (  0, 210, 230),   # CYAN
    "blowholes":            (  0, 210, 230),
    "porosity":             (  0, 210, 230),
    "surface porosity":     (  0, 210, 230),
    "undercut":             (255,  50,  50),   # RED
    "underfill":            ( 30, 120, 255),   # BLUE
    "underfill valleys":    ( 30, 120, 255),
    "excess reinforcement": (255,  50, 255),   # MAGENTA
    "excess reinforce":     (255,  50, 255),
    "high hump":            (255,  50, 255),
    "high humps":           (255,  50, 255),
}
DEFAULT_COLOR = (220, 220, 220)

# Only the 4 priority defect types shown in legend
LEGEND_ITEMS = [
    ("Blowholes / Porosity",  (  0, 210, 230)),
    ("Undercut",              (255,  50,  50)),
    ("Underfill",             ( 30, 120, 255)),
    ("Excess Reinforcement",  (255,  50, 255)),
]

# ── Priority defect keywords — anything NOT matching is skipped ───────────────
PRIORITY_KEYWORDS = (
    "blowhole", "porosity", "undercut", "underfill", "excess reinforcement",
    "excess reinforce", "high hump",
)

def _is_priority_defect(defect: Defect) -> bool:
    """Return True only for the 4 priority defect types."""
    t = defect.type.lower().strip().replace("_", " ")
    return any(kw in t or t in kw for kw in PRIORITY_KEYWORDS)


# Shape routing — which visual shape to use for each defect
def _get_shape(defect: Defect) -> str:
    """Return 'circle', 'oval', or 'rectangle' based on defect type."""
    t = defect.type.lower().strip().replace("_", " ")
    if any(kw in t for kw in ("blowhole", "porosity")):
        return "circle"
    if any(kw in t for kw in ("excess reinforcement", "excess reinforce", "high hump")):
        return "oval"
    # undercut, underfill → rectangle
    return "rectangle"

FONT_PATHS_BOLD = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf",
    "C:/Windows/Fonts/arialbd.ttf",
    "C:/Windows/Fonts/segoeuib.ttf",
]
FONT_PATHS_REG = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
    "C:/Windows/Fonts/arial.ttf",
    "C:/Windows/Fonts/segoeui.ttf",
]


def _load_font(paths, size):
    for p in paths:
        try:
            return ImageFont.truetype(p, size)
        except Exception:
            pass
    try:
        return ImageFont.load_default(size=size)
    except Exception:
        return ImageFont.load_default()


def _text_size(draw: ImageDraw.ImageDraw, text: str, font) -> Tuple[int, int]:
    try:
        tb = draw.textbbox((0, 0), text, font=font)
        return tb[2] - tb[0], tb[3] - tb[1]
    except Exception:
        return len(text) * 9, 16


def _get_color(defect: Defect) -> Tuple[int, int, int]:
    # Replace underscores with spaces so "excess_reinforcement" matches "excess reinforcement"
    t = defect.type.lower().strip().replace("_", " ")
    if t in DEFECT_COLORS:
        return DEFECT_COLORS[t]
    for keyword, color in DEFECT_COLORS.items():
        if keyword in t or t in keyword:
            return color
    return DEFAULT_COLOR


# ─── Header banner ────────────────────────────────────────────────────────────

def _build_header(w: int, overall_result: str, defects: List[Defect], scale: float) -> Image.Image:
    """
    Top dark banner styled like the sample image:
      - Left: "WELD INSPECTION STATUS: FAIL — SEVERE SURFACE DEFECTS DETECTED"
      - Right box: "DEFECTS FOUND: X, Y | NOT DETECTED: Z"
    """
    font_size_big = max(18, int(26 * scale))
    font_size_sm  = max(12, int(15 * scale))
    font_big  = _load_font(FONT_PATHS_BOLD, font_size_big)
    font_sm   = _load_font(FONT_PATHS_REG,  font_size_sm)

    bar_h = max(60, int(72 * scale))
    result_upper = overall_result.upper()

    if result_upper == "PASS":
        bg_color   = (0, 120, 0)
        status_txt = f"WELD INSPECTION STATUS: PASS — NO CRITICAL DEFECTS DETECTED"
    elif result_upper == "FAIL":
        bg_color   = (20, 20, 20)
        status_txt = f"WELD INSPECTION STATUS: FAIL — SEVERE SURFACE DEFECTS DETECTED"
    else:
        bg_color   = (60, 40, 0)
        status_txt = f"WELD INSPECTION STATUS: REVIEW — DEFECTS REQUIRE EVALUATION"

    img = Image.new("RGB", (w, bar_h), bg_color)
    draw = ImageDraw.Draw(img)

    # Colored left stripe
    stripe_color = (0, 200, 0) if result_upper == "PASS" else ((255, 50, 50) if result_upper == "FAIL" else (255, 160, 0))
    draw.rectangle([0, 0, max(6, int(8*scale)), bar_h], fill=stripe_color)

    # Status text (left-aligned)
    pad = max(14, int(16 * scale))
    tw, th = _text_size(draw, status_txt, font_big)
    ty = (bar_h - th) // 2
    # Green highlight box around the text for PASS, red for FAIL
    highlight_color = (0, 160, 0) if result_upper == "PASS" else ((180, 0, 0) if result_upper == "FAIL" else (160, 100, 0))
    draw.rectangle([pad - 4, ty - 4, pad + tw + 8, ty + th + 4], fill=highlight_color)
    draw.text((pad, ty), status_txt, font=font_big, fill=(255, 255, 255))

    # Right box: defect list
    found_types = list({d.type for d in defects if d.bounding_box})
    all_types   = list({d.type for d in defects})
    not_found   = [t for t in all_types if t not in found_types]

    right_lines = []
    if found_types:
        right_lines.append("DEFECTS FOUND: " + ", ".join(sorted(set(found_types))))
    if not_found:
        right_lines.append("NOT DETECTED: " + ", ".join(sorted(set(not_found))))
    if not right_lines:
        right_lines = ["DEFECTS FOUND: None"]

    box_pad = max(6, int(8 * scale))
    max_tw = max((_text_size(draw, l, font_sm)[0] for l in right_lines), default=100)
    box_w  = max_tw + box_pad * 2
    box_h  = sum(_text_size(draw, l, font_sm)[1] + 4 for l in right_lines) + box_pad * 2
    bx1 = w - box_w - pad
    by1 = (bar_h - box_h) // 2

    draw.rectangle([bx1 - 2, by1 - 2, bx1 + box_w + 2, by1 + box_h + 2],
                   fill=(0, 0, 0), outline=(200, 200, 200), width=1)
    cy = by1 + box_pad
    for line in right_lines:
        draw.text((bx1 + box_pad, cy), line, font=font_sm, fill=(240, 240, 240))
        cy += _text_size(draw, line, font_sm)[1] + 4

    return img


# ─── Legend bar ───────────────────────────────────────────────────────────────

def _build_legend(defects: List[Defect], w: int, scale: float) -> Image.Image:
    font_size = max(13, int(17 * scale))
    font = _load_font(FONT_PATHS_BOLD, font_size)
    title_font = _load_font(FONT_PATHS_BOLD, max(11, int(13 * scale)))

    items = LEGEND_ITEMS

    tmp = Image.new("RGB", (1, 1))
    tmp_d = ImageDraw.Draw(tmp)

    pad = max(8, int(10 * scale))
    swatch = max(16, int(20 * scale))

    item_widths = [swatch + 8 + _text_size(tmp_d, n, font)[0] + pad * 2 for n, _ in items]
    _, th = _text_size(tmp_d, "Ag", font)
    _, tth = _text_size(tmp_d, "DEFECT LEGEND:", title_font)
    bar_h = tth + th + pad * 3 + 6

    bar = Image.new("RGB", (w, bar_h), (12, 12, 22))
    d   = ImageDraw.Draw(bar)
    d.line([(0, 0), (w, 0)], fill=(80, 120, 200), width=2)

    d.text((pad, 4), "DEFECT LEGEND:", font=title_font, fill=(180, 200, 255))
    cy = 4 + tth + 4
    cx = pad

    for (name, color), iw in zip(items, item_widths):
        if cx + iw > w - pad:
            cx = pad
            cy += swatch + 6
        # Colored swatch
        d.rectangle([cx, cy, cx + swatch, cy + swatch], fill=color, outline=(0,0,0), width=1)
        lw, lh = _text_size(d, name, font)
        d.text((cx + swatch + 6, cy + (swatch - lh) // 2), name, font=font, fill=color)
        cx += iw

    return bar


# ─── Main defect drawing ──────────────────────────────────────────────────────

def _draw_defect_shape(
    draw: ImageDraw.ImageDraw,
    x1: int, y1: int, x2: int, y2: int,
    color: Tuple[int, int, int],
    line_w: int,
    shape: str,
):
    """
    Draw the defect marker using the shape appropriate for its type:
      - 'circle'    → perfect circle (uses the bounding box's inscribed circle)
      - 'oval'      → ellipse fitted to the full bounding box
      - 'rectangle' → standard rectangle (undercut, underfill)
    Each shape has a black outer shadow for contrast.
    """
    if shape == "circle":
        # Inscribed circle centred on the bounding box
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        r  = max(8, min((x2 - x1), (y2 - y1)) // 2)
        draw.ellipse([cx - r, cy - r, cx + r, cy + r],
                     outline=color, width=line_w)
    elif shape == "oval":
        # Ellipse fitted to the full bounding box
        draw.ellipse([x1, y1, x2, y2],
                     outline=color, width=line_w)
    else:
        # Rectangle (default)
        draw.rectangle([x1, y1, x2, y2],
                       outline=color, width=line_w)


def _draw_label_with_arrow(
    draw: ImageDraw.ImageDraw,
    label: str,
    box_cx: int, box_cy: int,
    box_x1: int, box_y1: int, box_x2: int, box_y2: int,
    color: Tuple[int,int,int],
    font,
    img_w: int, img_h: int,
    label_idx: int,
    total_labels: int,
    scale: float,
):
    """
    Draw the label text in a filled pill ABOVE or BELOW the box (alternating),
    with an arrow line from the pill to the box edge.
    Similar to the sample image style.
    """
    tw, th = _text_size(draw, label, font)
    pad = max(5, int(6 * scale))
    pill_w = tw + pad * 2
    pill_h = th + pad * 2

    # Decide label placement: alternate above/below
    arrow_margin = max(20, int(30 * scale))
    if label_idx % 2 == 0:
        # Label ABOVE the box
        label_y = box_y1 - arrow_margin - pill_h
        if label_y < 4:
            label_y = box_y2 + arrow_margin
    else:
        # Label BELOW the box
        label_y = box_y2 + arrow_margin
        if label_y + pill_h > img_h - 4:
            label_y = box_y1 - arrow_margin - pill_h

    # Horizontal: centre label on box
    label_x = box_cx - pill_w // 2
    label_x = max(4, min(img_w - pill_w - 4, label_x))

    pill_cx = label_x + pill_w // 2
    pill_cy = label_y + pill_h // 2

    # Arrow endpoint on the box edge
    if label_y < box_y1:
        arrow_end = (box_cx, box_y1)
    else:
        arrow_end = (box_cx, box_y2)

    # Draw arrow line (black outline + colored)
    lw = max(2, int(3 * scale))
    draw.line([(pill_cx, label_y + pill_h if label_y < box_y1 else label_y),
               arrow_end], fill=(0, 0, 0), width=lw + 2)
    draw.line([(pill_cx, label_y + pill_h if label_y < box_y1 else label_y),
               arrow_end], fill=color, width=lw)

    # Arrowhead (small triangle)
    aex, aey = arrow_end
    _draw_arrowhead(draw, pill_cx, label_y + pill_h if label_y < box_y1 else label_y,
                    aex, aey, color, size=max(6, int(8 * scale)))

    # Filled pill background (dark with colored border)
    draw.rectangle([label_x - 1, label_y - 1, label_x + pill_w + 1, label_y + pill_h + 1],
                   fill=(0, 0, 0), outline=color, width=max(2, lw))
    draw.rectangle([label_x, label_y, label_x + pill_w, label_y + pill_h],
                   fill=(20, 20, 20))
    # Label text in the defect color
    draw.text((label_x + pad, label_y + pad), label, font=font, fill=color)


def _draw_arrowhead(draw, x1, y1, x2, y2, color, size=8):
    """Draw a small arrowhead at (x2,y2) pointing from (x1,y1)."""
    angle = math.atan2(y2 - y1, x2 - x1)
    spread = math.pi / 6
    pts = [
        (x2, y2),
        (x2 - size * math.cos(angle - spread), y2 - size * math.sin(angle - spread)),
        (x2 - size * math.cos(angle + spread), y2 - size * math.sin(angle + spread)),
    ]
    draw.polygon(pts, fill=color)


# ─── Scale Bar Drawing helper ──────────────────────────────────────────────────

def _get_font(size: int):
    return _load_font(FONT_PATHS_REG, size)


def append_0_20cm_scale_bar(image: Image.Image) -> Image.Image:
    w, h = image.size
    ruler_h = max(45, int(h * 0.07))
    canvas = Image.new("RGB", (w, h + ruler_h), (14, 18, 28))
    canvas.paste(image, (0, 0))
    draw = ImageDraw.Draw(canvas)
    
    # Blue colored measurement band across the top of ruler strip
    draw.rectangle([0, h, w, h + int(ruler_h * 0.35)], fill=(28, 80, 170))
    
    pad_x = int(w * 0.025)
    span_w = w - 2 * pad_x
    if span_w <= 0:
        return canvas
        
    font = _get_font(max(12, int(ruler_h * 0.28)))
    
    for cm in range(21):
        tx = int(pad_x + (cm / 20.0) * span_w)
        if cm % 2 == 0 or cm in (5, 15):
            draw.line([(tx, h), (tx, h + int(ruler_h * 0.55))], fill=(200, 220, 250), width=max(2, int(w * 0.002)))
            label = f"{cm} cm" if cm in (0, 5, 10, 15, 20) else f"{cm}"
            try:
                bbox = draw.textbbox((0, 0), label, font=font)
                tw = bbox[2] - bbox[0]
            except Exception:
                tw = int(len(label) * 6)
            draw.text((tx - tw // 2, h + int(ruler_h * 0.58)), label, fill=(220, 235, 255), font=font)
        else:
            draw.line([(tx, h), (tx, h + int(ruler_h * 0.38))], fill=(130, 160, 200), width=max(1, int(w * 0.001)))
            
    return canvas


# ─── Main public function ─────────────────────────────────────────────────────

def annotate_image(
    image_bytes: bytes,
    defects: List[Defect],
    overall_result: str = "review",
) -> bytes:
    """
    Draw professional defect annotations on the stitched weld image.

    Only the 4 priority defect types are drawn and shown in the legend:
      Blowhole → circle | Excess Reinforcement → oval | Undercut/Underfill → rectangle

    Layout:
      1. Dark header banner (status + defect counts)
      2. Weld image with color-coded shapes per defect type
      3. Bottom legend bar

    Returns:
        JPEG bytes of the annotated image.
    """
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        w, h = img.size

        scale     = max(0.5, min(3.5, w / 1600))
        line_w    = max(3, int(5 * scale))
        font_size = max(13, int(17 * scale))
        font_lbl  = _load_font(FONT_PATHS_BOLD, font_size)

        # ── Filter to priority defects FIRST ──────────────────────────────────
        # This ensures header counts, legend entries, and drawn shapes are
        # always in sync. Non-priority types (spatter, crack, etc.) are ignored.
        priority_defects = [d for d in defects if _is_priority_defect(d)]
        labeled_defects  = [d for d in priority_defects if d.bounding_box]

        logger.info(
            f"[annotate] Total defects: {len(defects)} | "
            f"Priority: {len(priority_defects)} | Drawable: {len(labeled_defects)}"
        )

        # ── Draw defect shapes on the weld image ──────────────────────────────
        draw = ImageDraw.Draw(img)

        for defect in labeled_defects:
            bb    = defect.bounding_box
            color = _get_color(defect)
            shape = _get_shape(defect)

            # Handle both 0.0–1.0 and 0–1000 coordinate scales
            bb_x = bb.x     / 1000.0 if bb.x     > 1.0 else bb.x
            bb_y = bb.y     / 1000.0 if bb.y     > 1.0 else bb.y
            bb_w = bb.width / 1000.0 if bb.width > 1.0 else bb.width
            bb_h = bb.height/ 1000.0 if bb.height> 1.0 else bb.height

            x1 = max(0,     int(bb_x          * w))
            y1 = max(0,     int(bb_y          * h))
            x2 = min(w - 1, int((bb_x + bb_w) * w))
            y2 = min(h - 1, int((bb_y + bb_h) * h))

            # Enforce minimum visible size
            if x2 - x1 < 10: x2 = min(w - 1, x1 + 10)
            if y2 - y1 < 8:  y2 = min(h - 1, y1 + 8)

            _draw_defect_shape(draw, x1, y1, x2, y2, color, line_w, shape)

        # ── Compose: weld image + legend bar (no top banner) ──────────────────
        legend  = _build_legend(priority_defects, w, scale)
        total_h = h + legend.height

        final = Image.new("RGB", (w, total_h), (10, 10, 20))
        final.paste(img,    (0, 0))
        final.paste(legend, (0, h))

        buf = io.BytesIO()
        final.save(buf, format="JPEG", quality=94)
        return buf.getvalue()

    except Exception as e:
        logger.error(f"Annotation failed: {e}", exc_info=True)
        return image_bytes

