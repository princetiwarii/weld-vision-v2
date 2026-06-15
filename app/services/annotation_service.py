"""
annotation_service.py
=====================
Draws defect annotations on weld images to match the Tata Steel / CWI
reference inspection style shown in sample images.

Reference style (from sample images):
  - BOLD colored labels with black text outline — clearly readable on any background
  - Arrows pointing FROM the label TO the defect shape
  - Shapes drawn tightly over the defect region:
      Blowholes   → Small circles with center dot (red), one per visible pore
      Undercut    → Continuous wavy/straight line along the weld toe (orange)
      Underfill   → Curved bracket / arch below the weld centerline (blue)
      Excess Rein → Bumpy/irregular polygon outline around the hump (purple)
      Spatter     → Small filled dots scattered in the spatter region (green)
      Overlap     → Dashed line along overlap zone (grey)
      Other       → Tight rectangle outline (grey)
  - Legend box bottom-right with dark background
  - Labels staggered above/below to avoid overlap
  - Font size scales with image width for readability

Color coding (strict):
  Blowholes / Porosity  → Red    #FF0000
  Undercut              → Orange #FF8C00
  Underfill             → Blue   #1E90FF
  Excess Reinforcement  → Purple #9400D3
  Spatter               → Green  #00CC00
  Overlap               → Grey   #C0C0C0
  Other / Unknown       → LightGrey #AAAAAA
"""
from PIL import Image, ImageDraw, ImageFont
import io
import math
import random
from typing import List, Tuple, Optional
from app.schemas.inspection import Defect, DefectSeverity
from loguru import logger

# ─── Color mapping ────────────────────────────────────────────────────────────
DEFECT_TYPE_COLORS: dict = {
    "blowhole":              (255,   0,   0),   # Red
    "blowholes":             (255,   0,   0),
    "porosity":              (255,   0,   0),
    "surface porosity":      (255,   0,   0),
    "undercut":              (255, 140,   0),   # Dark Orange
    "underfill":             ( 30, 144, 255),   # Dodger Blue
    "underfill valleys":     ( 30, 144, 255),
    "notable underfill":     ( 30, 144, 255),
    "excess reinforcement":  (148,   0, 211),   # Dark Violet / Purple
    "excess reinforce":      (148,   0, 211),
    "high hump":             (148,   0, 211),
    "high humps":            (148,   0, 211),
    "spatter":               (  0, 204,   0),   # Green
    "widespread spatter":    (  0, 204,   0),
    "overlap":               (192, 192, 192),   # Silver / Grey
    "lack of fusion":        (255, 215,   0),   # Gold / Yellow
    "lack of penetration":   (255, 215,   0),
    "burn-through":          (255,  69,   0),   # OrangeRed
    "crack":                 (220,  20,  60),   # Crimson
    "cracks":                (220,  20,  60),
    "arc strike":            (255, 140,   0),
    "slag":                  (160,  82,  45),
}
DEFAULT_COLOR = (170, 170, 170)

LEGEND_ITEMS = [
    ("Red    = Blowholes",             (255,   0,   0)),
    ("Orange = Undercut",              (255, 140,   0)),
    ("Blue   = Underfill",             ( 30, 144, 255)),
    ("Purple = Excess Reinforcement",  (148,   0, 211)),
    ("Green  = Spatter",               (  0, 204,   0)),
    ("Yellow = Lack of Fusion",        (255, 215,   0)),
    ("Grey   = Other Defects",         (170, 170, 170)),
]

FONT_PATHS_BOLD = [
    "C:/Windows/Fonts/arialbd.ttf",
    "C:/Windows/Fonts/segoeuib.ttf",
    "C:/Windows/Fonts/calibrib.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf",
]
FONT_PATHS_REG = [
    "C:/Windows/Fonts/arial.ttf",
    "C:/Windows/Fonts/segoeui.ttf",
    "C:/Windows/Fonts/calibri.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
]


def _load_font(paths, size):
    for p in paths:
        try:
            return ImageFont.truetype(p, size)
        except Exception:
            pass
    return ImageFont.load_default()


def _fonts(scale: float = 1.0):
    """Return (label_font, small_font) scaled to image size."""
    label_size = max(16, int(24 * scale))
    small_size = max(12, int(17 * scale))
    return (
        _load_font(FONT_PATHS_BOLD, label_size),
        _load_font(FONT_PATHS_REG,  small_size),
    )


def _get_defect_color(defect: Defect) -> Tuple[int, int, int]:
    t = defect.type.lower().strip()
    # Exact match first
    if t in DEFECT_TYPE_COLORS:
        return DEFECT_TYPE_COLORS[t]
    # Substring match
    for keyword, color in DEFECT_TYPE_COLORS.items():
        if keyword in t or t in keyword:
            return color
    return DEFAULT_COLOR


# ─── Drawing primitives ───────────────────────────────────────────────────────

def _draw_outlined_text(
    draw: ImageDraw.ImageDraw,
    xy: Tuple[int, int],
    text: str,
    font,
    fill: Tuple,
    outline_width: int = 2,
    outline_color: Tuple = (0, 0, 0),
):
    """Text with thick black outline so it reads on any background."""
    x, y = xy
    for dx in range(-outline_width, outline_width + 1):
        for dy in range(-outline_width, outline_width + 1):
            if dx != 0 or dy != 0:
                draw.text((x + dx, y + dy), text, fill=outline_color, font=font)
    draw.text(xy, text, fill=fill, font=font)


def _text_size(draw: ImageDraw.ImageDraw, text: str, font) -> Tuple[int, int]:
    try:
        tb = draw.textbbox((0, 0), text, font=font)
        return tb[2] - tb[0], tb[3] - tb[1]
    except Exception:
        return len(text) * 10, 18


def _draw_arrow(
    draw: ImageDraw.ImageDraw,
    from_pt: Tuple[int, int],
    to_pt: Tuple[int, int],
    color: Tuple,
    line_width: int = 2,
    arrowhead_size: int = 10,
):
    """Draw a line with an arrowhead pointing at to_pt."""
    fx, fy = from_pt
    tx, ty = to_pt
    draw.line([(fx, fy), (tx, ty)], fill=(0, 0, 0), width=line_width + 2)
    draw.line([(fx, fy), (tx, ty)], fill=color, width=line_width)

    # Arrowhead
    dist = math.hypot(tx - fx, ty - fy)
    if dist < 1:
        return
    ux = (tx - fx) / dist
    uy = (ty - fy) / dist
    # Perpendicular
    px = -uy
    py = ux
    ah = arrowhead_size
    pts = [
        (tx, ty),
        (tx - ah * ux + (ah * 0.4) * px, ty - ah * uy + (ah * 0.4) * py),
        (tx - ah * ux - (ah * 0.4) * px, ty - ah * uy - (ah * 0.4) * py),
    ]
    draw.polygon(pts, fill=color, outline=(0, 0, 0))


def _draw_wavy_line(draw, p1, p2, color, width=3, amplitude=6, frequency=0.02):
    """Wavy / sine-wave line — used for Undercut along weld toe."""
    x1, y1 = p1
    x2, y2 = p2
    dist = math.hypot(x2 - x1, y2 - y1)
    if dist <= 0:
        return
    steps = max(30, int(dist))
    nx = -(y2 - y1) / dist
    ny = (x2 - x1) / dist
    pts = []
    for i in range(steps + 1):
        t = i / steps
        bx = x1 + t * (x2 - x1)
        by = y1 + t * (y2 - y1)
        off = math.sin(i * frequency * 2 * math.pi) * amplitude
        pts.append((bx + nx * off, by + ny * off))
    draw.line(pts, fill=(0, 0, 0), width=width + 3, joint="curve")
    draw.line(pts, fill=color, width=width, joint="curve")


def _draw_bumpy_polygon(draw, x1, y1, x2, y2, color, width=3, bumps=12, amplitude=8):
    """Bumpy / irregular closed polygon — used for Excess Reinforcement humps."""
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    rx = max(6, (x2 - x1) / 2)
    ry = max(6, (y2 - y1) / 2)
    steps = max(60, bumps * 14)
    pts = []
    for i in range(steps + 1):
        angle = i * 2 * math.pi / steps
        # Absolute pixel offset rather than a ratio, so it's always visible
        offset = math.sin(angle * bumps) * amplitude
        px = cx + (rx + offset) * math.cos(angle)
        py = cy + (ry + offset) * math.sin(angle)
        pts.append((px, py))
    draw.line(pts, fill=(0, 0, 0), width=width + 3, joint="curve")
    draw.line(pts, fill=color, width=width, joint="curve")


def _draw_scalloped_bracket(draw, x1, y1, x2, y2, color, width=3):
    """Scalloped bracket / arch lines — used for Underfill valleys."""
    mid_x = (x1 + x2) / 2
    # Top arch
    pts_top = [(x1, y1), (mid_x, y1 + (y2 - y1) * 0.45), (x2, y1)]
    # Bottom arch
    pts_bot = [(x1, y2), (mid_x, y2 - (y2 - y1) * 0.45), (x2, y2)]
    for pts in (pts_top, pts_bot):
        draw.line(pts, fill=(0, 0, 0), width=width + 3, joint="curve")
        draw.line(pts, fill=color, width=width, joint="curve")
    # Vertical side caps
    for xv in (x1, x2):
        draw.line([(xv, y1), (xv, y2)], fill=color, width=width)


def _draw_blowhole_circles(draw, x1, y1, x2, y2, color, n, scale, defect_id):
    """Draw N evenly distributed small circles representing blowholes."""
    radius = max(7, int(9 * scale))
    dot_r  = max(2, int(3 * scale))
    rng = random.Random(hash(defect_id))
    bbox_w = max(1, x2 - x1 - radius * 2)
    bbox_h = max(1, y2 - y1 - radius * 2)
    centers = []
    for _ in range(n):
        cx_i = x1 + radius + rng.randint(0, bbox_w)
        cy_i = y1 + radius + rng.randint(0, bbox_h)
        # Black halo
        draw.ellipse(
            [cx_i - radius - 3, cy_i - radius - 3, cx_i + radius + 3, cy_i + radius + 3],
            outline=(0, 0, 0), width=2,
        )
        # Colored ring
        draw.ellipse(
            [cx_i - radius, cy_i - radius, cx_i + radius, cy_i + radius],
            outline=color + (255,), width=2,
        )
        # Center dot
        draw.ellipse(
            [cx_i - dot_r, cy_i - dot_r, cx_i + dot_r, cy_i + dot_r],
            fill=color + (255,),
        )
        centers.append((cx_i, cy_i))
    return centers


def _draw_spatter_dots(draw, x1, y1, x2, y2, color, n, scale, defect_id):
    """Draw N small filled dots representing spatter droplets."""
    dot_r = max(3, int(5 * scale))
    rng = random.Random(hash(defect_id))
    bbox_w = max(1, x2 - x1)
    bbox_h = max(1, y2 - y1)
    for _ in range(n):
        dx = x1 + rng.randint(0, bbox_w)
        dy = y1 + rng.randint(0, bbox_h)
        draw.ellipse([dx - dot_r - 1, dy - dot_r - 1, dx + dot_r + 1, dy + dot_r + 1],
                     fill=(0, 0, 0))
        draw.ellipse([dx - dot_r, dy - dot_r, dx + dot_r, dy + dot_r],
                     fill=color + (255,))


# ─── Label placement ──────────────────────────────────────────────────────────

class LabelPlacer:
    """Tracks placed labels and finds non-overlapping positions."""

    def __init__(self, img_w: int, img_h: int):
        self.img_w = img_w
        self.img_h = img_h
        self.placed: List[Tuple[int, int, int, int]] = []  # (x, y, w, h) rectangles

    def _overlaps(self, x, y, w, h, pad=8) -> bool:
        for px, py, pw, ph in self.placed:
            if (x - pad < px + pw and x + w + pad > px and
                    y - pad < py + ph and y + h + pad > py):
                return True
        return False

    def find_position(
        self,
        anchor_x: int,
        anchor_y: int,
        tw: int,
        th: int,
        prefer_above: bool = True,
    ) -> Tuple[int, int]:
        """Find a non-overlapping label position using a dynamic outward search."""
        pad = 12
        
        # Step sizes based on actual text dimensions
        step_y = th + pad
        step_x = max(20, int(tw * 0.2))
        
        candidates = []
        
        # Generate candidates in an expanding grid
        for radius in range(0, 7):
            for dy_steps in range(-radius, radius + 1):
                for dx_steps in range(-radius, radius + 1):
                    # Only take the outer shell of the current radius
                    if abs(dy_steps) != radius and abs(dx_steps) != radius:
                        continue
                        
                    y_off = dy_steps * step_y
                    x_off = dx_steps * step_x
                    
                    # Base position centered over the offset point
                    lx = anchor_x + x_off - (tw // 2)
                    ly = anchor_y + y_off
                    
                    # Push away from the exact anchor center
                    if dy_steps == 0:
                        ly = anchor_y - step_y if prefer_above else anchor_y + step_y
                        
                    candidates.append((lx, ly))

        # Sort candidates by distance to anchor (penalize the non-preferred vertical direction)
        def score(pt):
            lx, ly = pt
            # Distance from label center to anchor
            dist = math.hypot(lx + tw//2 - anchor_x, ly + th//2 - anchor_y)
            # Add penalty if it's placed on the opposite side of preference
            if prefer_above and ly > anchor_y:
                dist += th * 3
            elif not prefer_above and ly < anchor_y:
                dist += th * 3
            return dist

        candidates.sort(key=score)

        for lx, ly in candidates:
            # Clamp to image boundaries
            lx = max(pad, min(self.img_w - tw - pad, lx))
            ly = max(pad, min(self.img_h - th - pad, ly))
            
            if not self._overlaps(lx, ly, tw, th, pad=pad):
                self.placed.append((lx, ly, tw, th))
                return lx, ly

        # Force place (last resort) - push it way up
        lx = max(pad, min(self.img_w - tw - pad, anchor_x - tw // 2))
        ly = max(pad, min(self.img_h - th - pad, anchor_y - step_y * 2))
        self.placed.append((lx, ly, tw, th))
        return lx, ly


# ─── Legend ───────────────────────────────────────────────────────────────────

def _draw_legend(draw: ImageDraw.ImageDraw, img_w: int, img_h: int, font, defects: List[Defect]):
    if not defects:
        items = [("No defects detected", (0, 200, 0))]
    else:
        present_colors = {_get_defect_color(d) for d in defects}
        items = [(name, col) for name, col in LEGEND_ITEMS if col in present_colors]
        if not items:
            items = [("Unknown defects detected", DEFAULT_COLOR)]

    tw_max = max((_text_size(draw, name, font)[0] for name, _ in items), default=200)
    line_h = _text_size(draw, "Defect Legend", font)[1] + 8

    pad = 10
    swatch_w = 22
    box_w = tw_max + swatch_w + pad * 3
    box_h = line_h * (len(items) + 1) + pad * 2
    bx = img_w - box_w - 14
    by = img_h - box_h - 14

    # Dark semi-transparent background
    draw.rectangle([bx - 3, by - 3, bx + box_w + 3, by + box_h + 3],
                   fill=(8, 8, 14), outline=(160, 160, 200), width=2)

    # Title
    _draw_outlined_text(draw, (bx + pad, by + pad), "Defect Legend", font=font,
                        fill=(220, 220, 245), outline_width=1)
    cy = by + pad + line_h

    for name, color in items:
        draw.rectangle([bx + pad, cy + 4, bx + pad + swatch_w, cy + line_h - 4],
                       fill=color, outline=(0, 0, 0))
        _draw_outlined_text(draw, (bx + pad + swatch_w + 6, cy), name, font=font,
                            fill=color, outline_width=1)
        cy += line_h


# ─── Main entry point ─────────────────────────────────────────────────────────

def annotate_image(image_bytes: bytes, defects: List[Defect]) -> bytes:
    """
    Draw exact shape overlays for defects (wavy lines, bumpy polygons) with a professional, structured label system.
    Labels are drawn as solid tags with thin pointers connecting them to the shapes, preventing visual clutter.
    """
    try:
        img  = Image.open(io.BytesIO(image_bytes)).convert("RGBA")
        w, h = img.size

        # Scale everything relative to image width (baseline 1200px)
        scale = max(0.5, min(3.5, w / 1200))
        line_w = max(2, int(3 * scale))
        label_font, small_font = _fonts(scale)

        overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
        draw    = ImageDraw.Draw(overlay)

        placer = LabelPlacer(w, h)

        # Group defects by type to prevent label spam
        from collections import defaultdict
        grouped_defects = defaultdict(list)
        for d in defects:
            grouped_defects[d.type.lower().strip()].append(d)

        for i, (t, defect_group) in enumerate(grouped_defects.items()):
            if not defect_group:
                continue

            # Use the first defect for color and general label formatting
            primary_defect = defect_group[0]
            color     = _get_defect_color(primary_defect)
            color255  = color + (255,)

            # Build a consolidated label text
            label_txt = primary_defect.label or primary_defect.type
            total_est = 0
            has_est = False
            for d in defect_group:
                if d.estimated_count:
                    import re as _re
                    nums = _re.findall(r'\d+', d.estimated_count)
                    if nums:
                        total_est += int(nums[0])
                        has_est = True
            
            if has_est and str(total_est) not in label_txt:
                label_txt = f"{label_txt} (~{total_est})"
            elif len(defect_group) > 1 and str(len(defect_group)) not in label_txt:
                 label_txt = f"{label_txt} ({len(defect_group)} instances)"
            
            if len(label_txt) > 55:
                label_txt = label_txt[:52] + "…"

            # We will pick the largest bounding box as the main anchor for the single label
            best_area = -1
            best_anchor = None
            best_bb_cx = 0

            # Draw all shapes for this group
            for defect in defect_group:
                bb = defect.bounding_box
                if not bb:
                    continue

                x1 = max(0, int(bb.x * w))
                y1 = max(0, int(bb.y * h))
                x2 = min(w - 1, int((bb.x + bb.width) * w))
                y2 = min(h - 1, int((bb.y + bb.height) * h))
                
                if x2 - x1 < 10: x2 = min(w - 1, x1 + 10)
                if y2 - y1 < 10: y2 = min(h - 1, y1 + 10)

                bb_cx = (x1 + x2) // 2
                bb_cy = (y1 + y2) // 2
                arrow_target = (bb_cx, bb_cy)

                # ─────────────────────────────────────────────────────────────────
                # Draw EXACT Defect Shapes
                # ─────────────────────────────────────────────────────────────────
                if "undercut" in t:
                    _draw_wavy_line(draw, (x1, y1), (x2, y1), color255,
                                    width=line_w, amplitude=max(2, int(4 * scale)))
                    if "bottom" in (defect.position or "").lower() or "both" in (defect.position or "").lower():
                        _draw_wavy_line(draw, (x1, y2), (x2, y2), color255,
                                        width=line_w, amplitude=max(2, int(4 * scale)))
                    arrow_target = (bb_cx, y1)

                elif "underfill" in t or "valley" in t:
                    _draw_scalloped_bracket(draw, x1, y1, x2, y2, color255, width=line_w + 1)
                    arrow_target = (bb_cx, (y1 + y2) // 2)

                elif "excess" in t or "reinforcement" in t or "hump" in t:
                    _draw_bumpy_polygon(draw, x1, y1, x2, y2, color255,
                                        width=line_w, bumps=12, amplitude=max(3, int(6 * scale)))
                    arrow_target = (bb_cx, y1)

                elif "blowhole" in t or "poros" in t:
                    import re as _re
                    n = 5
                    if defect.estimated_count:
                        nums = _re.findall(r'\d+', defect.estimated_count)
                        if nums: n = min(25, max(1, int(nums[0])))
                    centers = _draw_blowhole_circles(draw, x1, y1, x2, y2, color, n, scale, defect.defect_id)
                    arrow_target = centers[0] if centers else (bb_cx, bb_cy)

                elif "spatter" in t:
                    import re as _re
                    n = 20
                    if defect.estimated_count:
                        nums = _re.findall(r'\d+', defect.estimated_count)
                        if nums: n = min(80, max(5, int(nums[0])))
                    _draw_spatter_dots(draw, x1, y1, x2, y2, color, n, scale, defect.defect_id)
                    arrow_target = (bb_cx, bb_cy)

                elif "fusion" in t or "penetration" in t:
                    _draw_wavy_line(draw, (x1, bb_cy), (x2, bb_cy), color255,
                                    width=line_w, amplitude=max(2, int(3 * scale)), frequency=0.06)
                    arrow_target = (bb_cx, bb_cy)

                elif "overlap" in t:
                    seg = 18
                    gap = 8
                    cx_ = x1
                    while cx_ < x2:
                        ex = min(cx_ + seg, x2)
                        draw.line([(cx_, bb_cy), (ex, bb_cy)], fill=color255, width=line_w + 1)
                        cx_ += seg + gap
                    arrow_target = (bb_cx, bb_cy)
                else:
                    draw.ellipse([x1, y1, x2, y2], outline=color255, width=line_w)
                    arrow_target = (bb_cx, y1)

                # Check if this is the largest region for anchoring the label
                area = (x2 - x1) * (y2 - y1)
                if area > best_area:
                    best_area = area
                    best_anchor = arrow_target
                    best_bb_cx = bb_cx

            # ─────────────────────────────────────────────────────────────────
            # Draw ONE Label per defect group
            # ─────────────────────────────────────────────────────────────────
            if best_anchor:
                tw, th = _text_size(draw, label_txt, label_font)
                pad_x, pad_y = int(8 * scale), int(4 * scale)
                tag_w = tw + pad_x * 2
                tag_h = th + pad_y * 2
                
                prefer_above = (best_anchor[1] > h * 0.5)
                lx, ly = placer.find_position(best_bb_cx, best_anchor[1], tag_w, tag_h, prefer_above=prefer_above)

                # Pointer line to the main/largest shape
                tag_cx = lx + tag_w // 2
                tag_cy = ly + tag_h // 2
                draw.line([(tag_cx, tag_cy), best_anchor], fill=color + (200,), width=max(1, int(2 * scale)))

                # Tag Background
                draw.rectangle([lx, ly, lx + tag_w, ly + tag_h], fill=color + (230,))
                
                # Tag Text
                _draw_outlined_text(draw, (lx + pad_x, ly + pad_y), label_txt, font=label_font, fill=(255, 255, 255, 255), outline_width=1)
            elif not defect_group[0].bounding_box:
                # Fallback if no bounding box for any defect in the group
                lx, ly = 20, 20 + i * int(34 * scale) # Note: 'i' is not available here, but we can just use 20 for simplicity
                _draw_outlined_text(draw, (lx, ly), label_txt, font=label_font, fill=color255, outline_width=2)

        # Composite everything
        composited  = Image.alpha_composite(img, overlay)
        final_rgb   = composited.convert("RGB")

        buf = io.BytesIO()
        final_rgb.save(buf, format="JPEG", quality=93)
        return buf.getvalue()

    except Exception as e:
        logger.error(f"Annotation failed: {e}", exc_info=True)
        return image_bytes
