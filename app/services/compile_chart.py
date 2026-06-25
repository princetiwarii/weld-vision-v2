"""
compile_chart.py
================
Builds a vertically stacked 2-column inspection report card grid.

Reference style (from sample Image 3):
  - Dark background (#0E0F16)
  - Thin page header with title + segment count + pass/fail summary
  - 2-column card grid — each card:
      [Pair label strip]  — dark blue bg, white label left, PASS/FAIL score right
      [Annotated weld image] — fills full card width, TALL enough to be readable
      [Defect info strip] — lists defect types found in this pair
      [Physical scale ruler] — cm bar
  - Footer with timestamp

Key fix from reference: weld images must be large and readable, not squished.
Cards are wide landscape (weld images are 3:1 to 6:1 aspect ratio).
"""

from PIL import Image, ImageDraw, ImageFont
import io
import math
from typing import List, Optional

from app.schemas.inspection import FramePairResult, StatisticalSummary

# ── Layout constants ──────────────────────────────────────────────────────────
COLS        = 2          # 2-column grid
OUTER_PAD   = 28         # page margin
GUTTER      = 20         # gap between columns
LABEL_H     = 50         # pair-label strip height
DEFECT_H    = 60         # defect info strip height per card
RULER_H     = 56         # ruler strip height
HDR_H       = 84         # page header height
FOOTER_H    = 46

# Card image height — the most important setting.
# Weld stitched images are ~2:1 to 4:1 wide. At ~1650px card width, we
# want the image to be tall enough so weld details are clearly visible.
CARD_IMG_H  = 520        # px — increased from 400 for readability

PAGE_W      = 3508       # A4 @ 300 dpi landscape

# ── Colours ──────────────────────────────────────────────────────────────────
BG          = (10,  12,  20)
HDR_BG      = (16,  20,  36)
CARD_BG     = (18,  22,  34)
LABEL_BG    = (22,  28,  50)
DEFECT_BG   = (14,  18,  30)
RULER_BG    = (14,  18,  28)
RULER_FILL  = (28,  80, 170)
RULER_TICK  = (110, 138, 195)
RULER_TXT   = (180, 205, 235)
ACCENT      = ( 60, 140, 255)
WHITE       = (235, 238, 250)
GREY        = ( 90, 108, 135)
GREEN_OK    = ( 32, 190,  88)
RED_FAIL    = (210,  48,  48)
ORANGE_REV  = (210, 130,  30)
CARD_BORDER = ( 36,  44,  72)

# Defect type colors (must match annotation_service)
DEFECT_COLORS = {
    "blowhole":             (255,   0,   0),
    "porosity":             (255,   0,   0),
    "undercut":             (255, 140,   0),
    "underfill":            ( 30, 144, 255),
    "excess reinforcement": (148,   0, 211),
    "high hump":            (148,   0, 211),
    "spatter":              (  0, 204,   0),
    "overlap":              (192, 192, 192),
    "lack of fusion":       (255, 215,   0),
    "crack":                (220,  20,  60),
}


def _defect_color(defect_type: str) -> tuple:
    t = defect_type.lower()
    for k, c in DEFECT_COLORS.items():
        if k in t or t in k:
            return c
    return (150, 150, 150)


def _load_font(paths, size):
    for p in paths:
        try:
            return ImageFont.truetype(p, size)
        except Exception:
            pass
    return ImageFont.load_default()


def _fonts():
    bold = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        "C:/Windows/Fonts/arialbd.ttf",
        "C:/Windows/Fonts/segoeuib.ttf",
    ]
    reg = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/segoeui.ttf",
    ]
    return (
        _load_font(bold, 46),   # [0] page title
        _load_font(bold, 28),   # [1] card label bold
        _load_font(reg,  22),   # [2] ruler + defect info text
        _load_font(reg,  20),   # [3] footer
        _load_font(bold, 30),   # [4] PASS/FAIL score
    )


def _text_w(draw, text, font):
    try:
        tb = draw.textbbox((0, 0), text, font=font)
        return tb[2] - tb[0], tb[3] - tb[1]
    except Exception:
        return len(text) * 14, 22


def _result_color(result_val: str) -> tuple:
    v = result_val.lower()
    if v == "pass":   return GREEN_OK
    if v == "fail":   return RED_FAIL
    return ORANGE_REV


# ── Physical ruler ────────────────────────────────────────────────────────────
def _draw_ruler(draw, x, y, w, h, start_cm, end_cm, font):
    span = end_cm - start_cm
    if span <= 0:
        return

    draw.rectangle([x, y, x + w, y + h], fill=RULER_BG)
    draw.rectangle([x + 1, y + 1, x + w - 1, y + h // 2 - 1], fill=RULER_FILL)

    px_per_cm = w / span
    nice = [0.5, 1, 2, 5, 10, 20, 50, 100]
    tick_iv = nice[-1]
    for iv in nice:
        if (span / iv) <= (w / 65):
            tick_iv = iv
            break

    first = math.ceil(start_cm / tick_iv) * tick_iv
    drawn = set()
    cm = first
    while cm <= end_cm + 0.001:
        px = x + int((cm - start_cm) * px_per_cm)
        if x <= px <= x + w:
            draw.line([(px, y), (px, y + h - 1)], fill=RULER_TICK, width=2)
            label = f"{cm:.0f} cm" if cm == int(cm) else f"{cm:.1f} cm"
            draw.text((px + 4, y + h // 2 - 9), label, fill=RULER_TXT, font=font)
            drawn.add(round(cm, 3))
        cm = round(cm + tick_iv, 6)

    if round(start_cm, 3) not in drawn:
        draw.text((x + 6, y + h // 2 - 9), f"{start_cm:.0f} cm", fill=RULER_TXT, font=font)


# ── Defect info strip ─────────────────────────────────────────────────────────
def _draw_defect_strip(draw, x, y, w, h, res: FramePairResult, font):
    """Draw a compact defect summary strip under the weld image."""
    draw.rectangle([x, y, x + w, y + h], fill=DEFECT_BG)

    if not res.defects:
        txt = "✓  No defects detected"
        tw, th = _text_w(draw, txt, font)
        draw.text((x + 12, y + (h - th) // 2), txt, fill=GREEN_OK, font=font)
        return

    # Build compact defect tokens: colored text per defect type
    seen_types = {}
    for d in res.defects:
        t = d.type
        seen_types[t] = seen_types.get(t, 0) + 1

    cx = x + 12
    cy = y + (h - 22) // 2
    for dtype, cnt in seen_types.items():
        color = _defect_color(dtype)
        label = f"● {dtype}" + (f" ×{cnt}" if cnt > 1 else "")
        tw, th = _text_w(draw, label, font)
        if cx + tw > x + w - 10:
            break
        # Outline
        for dx2 in (-1, 0, 1):
            for dy2 in (-1, 0, 1):
                if dx2 or dy2:
                    draw.text((cx + dx2, cy + dy2), label, fill=(0, 0, 0), font=font)
        draw.text((cx, cy), label, fill=color, font=font)
        cx += tw + 24


# ── Main ──────────────────────────────────────────────────────────────────────
def build_compile_chart(
    annotated_images: List[bytes],
    results: List[FramePairResult],
    summary: StatisticalSummary,
    title: str = "WeldVision — Weld Inspection Report",
    pair_cm_ranges: Optional[List[tuple]] = None,
) -> bytes:
    """
    Compile annotated weld images into a 2-column card grid report.

    Each card (top to bottom):
      [pair-label strip]     — dark header, label + PASS/FAIL score
      [annotated weld image] — full card width, CARD_IMG_H tall
      [defect info strip]    — colored defect type tokens
      [physical scale ruler] — cm ticks

    Cards ordered chronologically (pair index = position along weld).
    """
    fonts = _fonts()
    title_f, label_f, info_f, footer_f, score_f = fonts

    n = max(len(annotated_images), 1)

    # ── Card geometry ─────────────────────────────────────────────────────────
    card_w     = (PAGE_W - OUTER_PAD * 2 - GUTTER * (COLS - 1)) // COLS
    card_h     = LABEL_H + CARD_IMG_H + DEFECT_H + RULER_H

    rows = math.ceil(n / COLS)
    total_h = HDR_H + OUTER_PAD + rows * card_h + (rows - 1) * GUTTER + OUTER_PAD + FOOTER_H

    canvas = Image.new("RGB", (PAGE_W, total_h), BG)
    draw   = ImageDraw.Draw(canvas)

    # ── Page header ───────────────────────────────────────────────────────────
    draw.rectangle([0, 0, PAGE_W, HDR_H], fill=HDR_BG)
    # Left accent stripe
    draw.rectangle([0, 0, 7, HDR_H], fill=ACCENT)
    # Title
    draw.text((OUTER_PAD + 16, (HDR_H - 46) // 2), title, fill=WHITE, font=title_f)
    # Right: summary counts
    info = (
        f"{n} segment{'s' if n != 1 else ''}  │  "
        f"Pass: {summary.pass_count}  Fail: {summary.fail_count}  Review: {summary.review_count}"
    )
    iw, _ = _text_w(draw, info, label_f)
    draw.text((PAGE_W - OUTER_PAD - iw, (HDR_H - 28) // 2), info, fill=ACCENT, font=label_f)
    draw.line([(0, HDR_H - 2), (PAGE_W, HDR_H - 2)], fill=ACCENT, width=2)

    y_start = HDR_H + OUTER_PAD

    # ── Card grid ─────────────────────────────────────────────────────────────
    for idx in range(n):
        col = idx % COLS
        row = idx // COLS

        cx = OUTER_PAD + col * (card_w + GUTTER)
        cy = y_start  + row * (card_h + GUTTER)

        img_bytes = annotated_images[idx] if idx < len(annotated_images) else None
        res       = results[idx] if idx < len(results) else None

        # Card background + border
        draw.rectangle([cx - 1, cy - 1, cx + card_w + 1, cy + card_h + 1],
                        fill=CARD_BORDER)
        draw.rectangle([cx, cy, cx + card_w, cy + card_h], fill=CARD_BG)

        # ── Label strip ───────────────────────────────────────────────────────
        draw.rectangle([cx, cy, cx + card_w, cy + LABEL_H], fill=LABEL_BG)

        if res:
            lbl = f"Pair {idx + 1}  |  {res.source_frame_a_label}"
            if res.source_frame_b_label:
                lbl += f" + {res.source_frame_b_label}"

            rc  = _result_color(res.overall_result.value)
            score_lbl = f"{res.overall_result.value.upper()}  {res.weld_quality_score:.0f}/100"

            lw, _ = _text_w(draw, lbl, label_f)
            sw, _ = _text_w(draw, score_lbl, score_f)

            draw.text((cx + 14, (LABEL_H - 28) // 2 + cy), lbl, fill=WHITE, font=label_f)
            # Score pill background
            sx = cx + card_w - sw - 16
            sy = cy + (LABEL_H - 30) // 2
            draw.rectangle([sx - 8, sy - 4, sx + sw + 8, sy + 34], fill=CARD_BG)
            draw.text((sx, sy), score_lbl, fill=rc, font=score_f)
        else:
            draw.text((cx + 14, (LABEL_H - 28) // 2 + cy),
                      f"Segment {idx + 1}", fill=GREY, font=label_f)

        img_y = cy + LABEL_H

        # ── Annotated weld image ──────────────────────────────────────────────
        if img_bytes:
            try:
                weld_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                orig_w, orig_h = weld_img.size
                # Resize to fill card width, maintain aspect ratio, then crop/pad to CARD_IMG_H
                target_w = card_w
                aspect = orig_h / orig_w if orig_w > 0 else 0.5
                scaled_h = int(target_w * aspect)

                if scaled_h >= CARD_IMG_H:
                    # Image is taller than card slot — scale down to CARD_IMG_H height and pad width
                    scaled_w = int(CARD_IMG_H * orig_w / orig_h) if orig_h > 0 else target_w
                    weld_img = weld_img.resize((scaled_w, CARD_IMG_H), Image.LANCZOS)
                    padded = Image.new("RGB", (card_w, CARD_IMG_H), (10, 12, 20))
                    padded.paste(weld_img, ((card_w - scaled_w) // 2, 0))
                    weld_img = padded
                else:
                    # Image is shorter than card slot — scale to CARD_IMG_H height
                    scaled_w = int(CARD_IMG_H * orig_w / orig_h) if orig_h > 0 else target_w
                    weld_img = weld_img.resize((scaled_w, CARD_IMG_H), Image.LANCZOS)
                    if scaled_w >= card_w:
                        # Crop to width
                        weld_img = weld_img.crop((0, 0, card_w, CARD_IMG_H))
                    else:
                        # Pad with dark background
                        padded = Image.new("RGB", (card_w, CARD_IMG_H), (10, 12, 20))
                        padded.paste(weld_img, ((card_w - scaled_w) // 2, 0))
                        weld_img = padded

                canvas.paste(weld_img, (cx, img_y))
            except Exception as e:
                logger.error(f"Card image failed: {e}")
                draw.rectangle([cx, img_y, cx + card_w, img_y + CARD_IMG_H], fill=(20, 24, 38))
                draw.text((cx + 20, img_y + CARD_IMG_H // 2 - 15),
                          "Image unavailable", fill=GREY, font=label_f)
        else:
            draw.rectangle([cx, img_y, cx + card_w, img_y + CARD_IMG_H], fill=(20, 24, 38))
            draw.text((cx + 20, img_y + CARD_IMG_H // 2 - 15),
                      "No image", fill=GREY, font=label_f)

        defect_y = img_y + CARD_IMG_H

        # ── Defect info strip ─────────────────────────────────────────────────
        if res:
            _draw_defect_strip(draw, cx, defect_y, card_w, DEFECT_H, res, info_f)
        else:
            draw.rectangle([cx, defect_y, cx + card_w, defect_y + DEFECT_H], fill=DEFECT_BG)

        ruler_y = defect_y + DEFECT_H

        # ── Ruler strip ───────────────────────────────────────────────────────
        has_range = (
            pair_cm_ranges
            and idx < len(pair_cm_ranges)
            and pair_cm_ranges[idx] is not None
        )
        if has_range:
            sc, ec = pair_cm_ranges[idx]
            _draw_ruler(draw, cx, ruler_y, card_w, RULER_H, sc, ec, info_f)
        else:
            draw.rectangle([cx, ruler_y, cx + card_w, ruler_y + RULER_H], fill=RULER_BG)
            draw.text((cx + 10, ruler_y + RULER_H // 2 - 9),
                      "No physical scale", fill=GREY, font=info_f)

    # ── Footer ────────────────────────────────────────────────────────────────
    fy = y_start + rows * card_h + (rows - 1) * GUTTER + OUTER_PAD // 2
    draw.line([(OUTER_PAD, fy), (PAGE_W - OUTER_PAD, fy)], fill=(36, 44, 70), width=1)

    from datetime import datetime, timezone
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    draw.text(
        (OUTER_PAD, fy + 12),
        f"WeldVision AI  •  Generated {ts}  •  Confidential",
        fill=GREY, font=footer_f,
    )

    buf = io.BytesIO()
    canvas.save(buf, format="JPEG", quality=94, optimize=True)
    return buf.getvalue()


# Import logger at module level
from loguru import logger
