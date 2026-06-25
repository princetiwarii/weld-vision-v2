"""
image_stitcher.py
=================
Stitch two frame JPEGs side-by-side into one image before sending
to Gemini.  This halves the number of API calls while giving the
model full context of adjacent weld sections.

If only one image is available (odd-total-frame edge case) the
single image is returned with its own scale bar.

A continuous physical scale bar is drawn at the bottom of the
stitched image so the viewer sees one ruler spanning the real-world
cm range (e.g. 20 cm – 40 cm) rather than two separate 0–20 scales.
"""
from PIL import Image, ImageDraw, ImageFont
import io
from loguru import logger

DIVIDER_WIDTH  = 4             # px separator between the two halves
DIVIDER_COLOR  = (60, 60, 60)  # dark grey

SCALE_H        = 28            # height of the scale-bar strip
SCALE_BG       = (20, 20, 28)
SCALE_TICK_COL = (180, 180, 200)
SCALE_TEXT_COL = (200, 200, 220)
SCALE_FILL_COL = (50, 120, 220)   # filled portion of the ruler


def _draw_scale_bar(
    draw: ImageDraw.ImageDraw,
    x: int,
    y: int,
    width: int,
    height: int,
    start_cm: float,
    end_cm: float,
):
    """
    Draw a horizontal scale bar from start_cm to end_cm inside the
    rectangle (x, y, x+width, y+height).
    """
    span_cm = end_cm - start_cm
    if span_cm <= 0:
        return

    # Background
    draw.rectangle([x, y, x + width, y + height], fill=SCALE_BG)

    # Filled colour band (full width — the entire strip represents the span)
    draw.rectangle([x + 1, y + 1, x + width - 1, y + height // 2], fill=SCALE_FILL_COL)

    # Choose a tick interval that keeps ticks readable
    nice_intervals = [0.5, 1, 2, 5, 10, 20, 50]
    px_per_cm      = width / span_cm
    target_ticks   = width / 60          # aim for ~1 tick per 60 px
    tick_interval  = nice_intervals[-1]
    for iv in nice_intervals:
        if span_cm / iv <= target_ticks * 1.5:
            tick_interval = iv
            break

    font_paths = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/segoeui.ttf",
    ]
    font = None
    for fp in font_paths:
        try:
            font = ImageFont.truetype(fp, 9)
            break
        except Exception:
            pass
    if font is None:
        font = ImageFont.load_default()

    cm = start_cm
    # Snap to the first tick at or above start_cm
    if start_cm % tick_interval != 0:
        cm = (int(start_cm / tick_interval) + 1) * tick_interval

    while cm <= end_cm + 0.001:
        px_offset = int((cm - start_cm) * px_per_cm)
        tx = x + px_offset
        if tx < x or tx > x + width:
            cm += tick_interval
            continue
        draw.line([(tx, y), (tx, y + height - 1)], fill=SCALE_TICK_COL, width=1)
        label = f"{cm:.0f}" if cm == int(cm) else f"{cm:.1f}"
        draw.text((tx + 2, y + height // 2 + 1), f"{label}cm", fill=SCALE_TEXT_COL, font=font)
        cm = round(cm + tick_interval, 6)

    # End-cap label
    draw.text((x + 2, y + height // 2 + 1), f"{start_cm:.1f}cm", fill=SCALE_TEXT_COL, font=font)


def stitch_pair(
    frame_a: bytes,
    frame_b: bytes | None,
    start_cm_a: float = 0.0,
    length_cm_a: float = 0.0,
    length_cm_b: float = 0.0,
) -> bytes:
    """
    Stitch frame_a and frame_b horizontally with a continuous scale bar.

    Parameters
    ----------
    frame_a      : JPEG/PNG bytes of the first frame
    frame_b      : JPEG/PNG bytes of the second frame (or None)
    start_cm_a   : physical start of frame_a on the weld (cumulative cm)
    length_cm_a  : physical length of frame_a in cm
    length_cm_b  : physical length of frame_b in cm (0 if no frame_b)
    """
    add_scale = (length_cm_a > 0)

    try:
        img_a = Image.open(io.BytesIO(frame_a)).convert("RGB")

        if frame_b is None:
            # Single frame — optionally add its own scale bar
            if add_scale:
                ha, wa = img_a.size[1], img_a.size[0]
                canvas = Image.new("RGB", (wa, ha + SCALE_H), SCALE_BG)
                canvas.paste(img_a, (0, 0))
                draw = ImageDraw.Draw(canvas)
                _draw_scale_bar(
                    draw, 0, ha, wa, SCALE_H,
                    start_cm_a, start_cm_a + length_cm_a,
                )
                buf = io.BytesIO()
                canvas.save(buf, format="JPEG", quality=90)
                return buf.getvalue()
            return frame_a

        img_b = Image.open(io.BytesIO(frame_b)).convert("RGB")

        # Normalise heights — resize B to match A's height
        ha, wa = img_a.size[1], img_a.size[0]
        hb, wb = img_b.size[1], img_b.size[0]

        if ha != hb:
            scale  = ha / hb
            new_wb = int(wb * scale)
            img_b  = img_b.resize((new_wb, ha), Image.LANCZOS)
            wb     = new_wb

        total_w = wa + DIVIDER_WIDTH + wb
        canvas_h = ha + (SCALE_H if add_scale else 0)

        canvas = Image.new("RGB", (total_w, canvas_h), DIVIDER_COLOR)
        canvas.paste(img_a, (0, 0))
        canvas.paste(img_b, (wa + DIVIDER_WIDTH, 0))

        if add_scale:
            draw = ImageDraw.Draw(canvas)

            # Scale bar spans the full stitched width but split at the divider
            # Left half: start_cm_a → start_cm_a + length_cm_a
            # Right half: start_cm_a + length_cm_a → start_cm_a + length_cm_a + length_cm_b
            end_cm_a = start_cm_a + length_cm_a
            end_cm_b = end_cm_a + length_cm_b

            # Draw as one continuous bar across both halves
            _draw_scale_bar(draw, 0, ha, wa, SCALE_H, start_cm_a, end_cm_a)
            _draw_scale_bar(draw, wa + DIVIDER_WIDTH, ha, wb, SCALE_H, end_cm_a, end_cm_b)

            # Draw a thin divider in the scale strip too
            draw.rectangle(
                [wa, ha, wa + DIVIDER_WIDTH, ha + SCALE_H],
                fill=DIVIDER_COLOR,
            )

        buf = io.BytesIO()
        canvas.save(buf, format="JPEG", quality=90)
        logger.debug(
            f"Stitched pair → {total_w}×{canvas_h}px  "
            f"[{start_cm_a:.1f}–{start_cm_a + length_cm_a + length_cm_b:.1f} cm]"
        )
        return buf.getvalue()

    except Exception as e:
        logger.error(f"Stitch failed, returning frame_a only: {e}")
        return frame_a
