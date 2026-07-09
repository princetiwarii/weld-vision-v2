"""
gemini_service.py
=================
Wraps Google Gemini Vision API for weld defect analysis.

Each call receives a **stitched image** (two consecutive frames
joined side-by-side). The scale/ruler visible in the image is used
as the ground-truth reference for ALL physical measurements.

Changes in this version:
  - WELD_MEASUREMENT_INSPECTION_PROMPT now demands the model read the
    scale bar and return total_weld_length_mm + per-defect measurements
    that are mathematically consistent with that scale.
  - inspect_with_measurements() builds a rich structured text output
    containing per-image measurements AND a summary table.
  - Annotation (generate-images) is handled entirely in annotation_service.py.
"""
from google import genai
from google.genai import types
import json
import re
import uuid
from fastapi import HTTPException, status
from app.core.config import settings
from app.schemas.inspection import (
    Defect, BoundingBox, WeldingStandardsCompliance,
    DefectSeverity, OverallResult, FramePairResult,
)
from loguru import logger

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------
WELD_INSPECTION_PROMPT = """\
Act as a Certified Welding Inspector (CWI).

Look in details where the defects actually ARE. Prioritize and explicitly look for the following critical defects: Underfill, Undercut, Blowholes (Porosity), and Excess Reinforcement. Also note other defects like spatter, lack of fusion, cracks, etc.
Mark them specifically by providing tight bounding boxes for each defect very clearly and neatly so our backend can draw them directly on the annotated stitched image.

CRITICAL: Bounding box coordinates must be precise decimals tightly clamped to the exact visible defect boundary (0.0 to 1.0 relative, where x=0 is left, y=0 is top, x=1 is right, y=1 is bottom).

Return the result EXACTLY in the JSON format as returning previously.

Required JSON format:
{
  "valid": true,
  "overall_result": "pass"|"fail"|"review",
  "weld_quality_score": 90,
  "defect_summary": {
    "Spatter": 2,
    "Undercut": 1
  },
  "defects": [
    {
      "defect_id": "Seq 1",
      "type": "<Defect Type>",
      "label": "<Descriptive Label e.g. 'Undercut' or 'Spatter'>",
      "description": "<Remarks>",
      "severity": "low"|"medium"|"high"|"critical",
      "estimated_count": "<Quantity if applicable>",
      "length_mm": 10.5,
      "bounding_box": {"x": 0.15, "y": 0.25, "width": 0.08, "height": 0.04},
      "location_description": "<Precise textual description of where the defect is located (e.g., 'Bottom toe of the weld on the right side')>"
    }
  ],
  "standards_compliance": [
    {"standard": "AWS D1.1", "compliant": false, "notes": "<Reason>"},
    {"standard": "ISO 5817", "compliant": false, "notes": "<Reason>"}
  ],
  "recommendations": [
    "<Actionable recommendation 1>"
  ],
  "model_notes": "<Any additional model notes or disclaimers>"
}
"""

WELD_BEAD_LOCALIZATION_PROMPT = """\
You are an expert computer vision system.
Your task is to locate the primary horizontal weld bead(s) in the image.
Provide a list of bounding boxes, where each box tightly encloses a weld bead segment.
Ignore the background. Focus only on the actual weld metal.

The image coordinate system:
- Coordinates must be a list of 4 integers [ymin, xmin, ymax, xmax] between 0 and 1000.
- ymin is the top edge, xmin is the left edge, ymax is the bottom edge, xmax is the right edge.

Return EXACTLY in this JSON format with no other text:
{
  "weld_beads": [
    [0, 0, 0, 0]
  ]
}
"""

WELD_DEFECT_LOCALIZATION_PROMPT = """\
You are an expert computer vision system analyzing a cropped image of a weld bead.
I have already identified a list of weld defects.
Your task is to locate each specific defect WITHIN THIS CROPPED IMAGE and provide its precise bounding box coordinates.

The image coordinate system:
- Coordinates must be a list of 4 integers [ymin, xmin, ymax, xmax] between 0 and 1000.
- ymin is the top edge, xmin is the left edge, ymax is the bottom edge, xmax is the right edge.
- The boxes must tightly enclose the visible defect area.

IMPORTANT:
- If a defect is NOT visible in this specific crop, set its bounding box values all to 0.
CRITICAL SIZING & POSITIONING RULES FOR PRIORITY DEFECTS:
- Our priority defects whose annotations we require on the image are ONLY: Underfill, Undercut, Excess Reinforcement, and Blowhole (Porosity).
- POSITIONING IS CRITICAL: The bounding box MUST BE LOCATED ON THE WELD BEAD ITSELF. DO NOT place bounding boxes floating out in the parent metal, rust, or background above or below the weld bead! For undercut and underfill, place the box directly on the toe/edge OF THE WELD BEAD.
- When locating these four defects, ensure the bounding box covers the WHOLE defect area completely so that when a transparent rectangle is drawn over it, the entire defect space is covered and visible.
- Every box MUST have minimum width/height corresponding to at least 60 units in the 0-1000 scale so it is clearly visible as a distinct rectangle drawn on the image — NOT a thin line or dot.

Defects to locate:
{defects_json}

Return EXACTLY in this JSON format with no other text:
{
  "defects": [
    {
      "defect_id": "<Exact defect_id from input>",
      "bounding_box": [0, 0, 0, 0]
    }
  ]
}
"""

WELD_MEASUREMENT_INSPECTION_PROMPT = """\
Act as an expert Certified Welding Inspector (CWI) performing a precision visual inspection.

I have provided a single weld image showing a 20 cm horizontal weld bead. A physical scale/ruler is visible at the bottom.

IMAGE LAYOUT (CRITICAL — read before anything else):
- The weld bead is the HORIZONTAL dark/silvery seam running LEFT to RIGHT across the image.
- The weld bead typically occupies the vertical band from approximately y=300 to y=700 on the vertical scale.
- ABOVE and BELOW the weld bead is parent/base metal — DO NOT place defect boxes there.
- The scale bar at the bottom spans 0 to 20 cm.

YOUR TASK — scan left to right:
1. READ the scale bar. Record total_weld_length_mm and scale_notes.

2. Scan the weld bead from LEFT to RIGHT. For each defect you find:
   - Identify what type it is (only these 4 types matter):
     1. Undercut
     2. Underfill
     3. Excess Reinforcement
     4. Blowhole (Porosity)
   - DO NOT report spatter, rust, discoloration, slag, or parent metal marks!

3. For each defect, provide a bounding_box that:
   - Is placed DIRECTLY ON THE WELD BEAD where the defect is visible.
   - The bounding_box must be a list of 4 integers [ymin, xmin, ymax, xmax] between 0 and 1000.
   - ymin is the top edge, xmin is the left edge, ymax is the bottom edge, xmax is the right edge.
   - The box must be large enough that when drawn as a transparent colored rectangle, the defect underneath is clearly highlighted and visible through it.

BOUNDING BOX RULES (MOST IMPORTANT):
- ALL boxes MUST sit ON the weld bead. The weld bead center is approximately ymin=450 to ymax=550.
- For undercut/underfill at the upper toe: ymin should be approximately 300-400, ymax ~420-580.
- For undercut/underfill at the lower toe: ymin should be approximately 550-650, ymax ~670-830.
- For excess reinforcement on the bead crown: ymin should be approximately 350-450, ymax ~500-650.
- For blowholes: place a box centered on the visible pit, minimum size 40x80.
- NEVER place a box above ymin=250 or below ymax=750 — those areas are parent metal, not weld.
- Maximum 10 defect entries total. Only report defects you are genuinely confident about (confidence >= 0.6).

4. NEVER MERGE separate defects into one box. Each continuous defect gets its own box.

5. For each defect also record:
   - "shape": "rectangle" for elongated defects (undercut, underfill, reinforcement), "circle" or "oval" for blowholes/pits.
   - "confidence": 0.0 to 1.0. Do NOT include defects below 0.6 confidence.
   - Length in mm along the weld axis using the scale bar.
   - Position: scale mark range (e.g. "3cm to 5cm"), location_side, weld_zone.

- location_side: "left" | "center" | "right" | "full-width"
- weld_zone: "upper_toe" | "bead_centre" | "lower_toe" | "heat_affected_zone" | "scattered"

Return EXACTLY this JSON, no other text:
{
  "valid": true,
  "overall_result": "pass"|"fail"|"review",
  "weld_quality_score": 85,
  "scale_detected": true,
  "scale_notes": "<e.g. Scale bar spans 0 cm to 20 cm, total 20 cm = 200 mm>",
  "total_weld_length_mm": 200.0,
  "defect_summary": {"Undercut": 2, "Underfill": 1},
  "defects": [
    {
      "defect_id": "Seq 1",
      "type": "<Defect Type>",
      "label": "<Short label e.g. 'Undercut at upper toe, 3-5 cm'>",
      "description": "<Detailed remarks>",
      "severity": "low"|"medium"|"high"|"critical",
      "confidence": 0.9,
      "shape": "rectangle"|"square"|"circle"|"oval",
      "estimated_count": "<e.g. 1>",
      "length_mm": 20.0,
      "width_mm": 2.0,
      "depth_mm": 0.0,
      "start_cm": 3.0,
      "end_cm": 5.0,
      "location_side": "left",
      "weld_zone": "upper_toe",
      "bounding_box": [350, 150, 470, 250],
      "location_description": "<Precise description>"
    }
  ],
  "standards_compliance": [
    {"standard": "AWS D1.1", "compliant": false, "notes": "<Reason>"},
    {"standard": "ISO 5817", "compliant": false, "notes": "<Reason>"}
  ],
  "recommendations": ["<Actionable recommendation 1>"],
  "model_notes": "<Any disclaimers>"
}
"""

_BBOX_SCHEMA = {
    "type": "ARRAY",
    "items": {
        "type": "INTEGER",
        "minimum": 0,
        "maximum": 1000
    },
    "minItems": 4,
    "maxItems": 4
}

_DEFECT_ENTRY_SCHEMA = {
    "type": "OBJECT",
    "property_ordering": [
        "defect_id", "type", "label", "confidence", "shape", "severity",
        "estimated_count", "length_mm", "width_mm", "depth_mm",
        "start_cm", "end_cm", "location_side", "weld_zone",
        "bounding_box", "location_description", "description",
    ],
    "properties": {
        "defect_id":             {"type": "STRING"},
        "type":                  {"type": "STRING"},
        "label":                 {"type": "STRING"},
        "confidence":            {"type": "NUMBER"},
        "shape":                 {"type": "STRING", "enum": ["rectangle", "square", "circle", "oval"]},
        "severity":              {"type": "STRING", "enum": ["low", "medium", "high", "critical"]},
        "estimated_count":       {"type": "STRING"},
        "length_mm":             {"type": "NUMBER"},
        "width_mm":              {"type": "NUMBER"},
        "depth_mm":              {"type": "NUMBER"},
        "start_cm":              {"type": "NUMBER"},
        "end_cm":                {"type": "NUMBER"},
        "location_side":         {"type": "STRING", "enum": ["left", "center", "right", "full-width"]},
        "weld_zone":             {"type": "STRING", "enum": ["upper_toe", "bead_centre", "lower_toe", "heat_affected_zone", "scattered"]},
        "bounding_box":          _BBOX_SCHEMA,
        "location_description":  {"type": "STRING"},
        "description":           {"type": "STRING"},
    },
    "required": ["defect_id", "type", "severity", "confidence", "shape", "bounding_box", "description"],
}

WELD_INSPECTION_RESPONSE_SCHEMA = {
    "type": "OBJECT",
    "property_ordering": [
        "valid", "overall_result", "weld_quality_score", "scale_detected",
        "scale_notes", "total_weld_length_mm", "defect_summary", "defects",
        "standards_compliance", "recommendations", "model_notes",
    ],
    "properties": {
        "valid":                 {"type": "BOOLEAN"},
        "overall_result":        {"type": "STRING", "enum": ["pass", "fail", "review"]},
        "weld_quality_score":    {"type": "NUMBER"},
        "scale_detected":        {"type": "BOOLEAN"},
        "scale_notes":           {"type": "STRING"},
        "total_weld_length_mm":  {"type": "NUMBER"},
        # Committing to counts-per-type BEFORE the itemized list (property_ordering
        # above forces this key to be generated first) is the count-consistency
        # cross-check described in the prompt.
        "defect_summary":        {"type": "OBJECT"},
        "defects":                {"type": "ARRAY", "items": _DEFECT_ENTRY_SCHEMA},
        "standards_compliance": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "standard":  {"type": "STRING"},
                    "grade":     {"type": "STRING"},
                    "compliant": {"type": "BOOLEAN"},
                    "notes":     {"type": "STRING"},
                },
                "required": ["standard", "compliant"],
            },
        },
        "recommendations": {"type": "ARRAY", "items": {"type": "STRING"}},
        "model_notes":     {"type": "STRING"},
    },
    "required": ["valid", "overall_result", "weld_quality_score", "defect_summary", "defects"],
}

WELD_BEAD_LOCALIZATION_RESPONSE_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "weld_beads": {
            "type": "ARRAY",
            "items": {
                "type": "ARRAY",
                "items": {"type": "INTEGER"},
                "minItems": 4,
                "maxItems": 4
            }
        }
    },
    "required": ["weld_beads"]
}

WELD_DEFECT_LOCALIZATION_RESPONSE_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "defects": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "defect_id": {"type": "STRING"},
                    "bounding_box": {
                        "type": "ARRAY",
                        "items": {"type": "INTEGER"},
                        "minItems": 4,
                        "maxItems": 4
                    }
                },
                "required": ["defect_id", "bounding_box"]
            }
        }
    },
    "required": ["defects"]
}


WELD_DEFECT_LOCALIZATION_FULL_IMAGE_PROMPT = """\
You are a precision computer vision system analyzing a weld inspection image.
I have already identified defects in this image. Your ONLY task is to draw a BOUNDING BOX around each defect.

This is the EXACT SAME full stitched image. Do NOT mentally crop or zoom.

CRITICAL — WELD BEAD LOCATION:
- The weld bead (the dark/silvery seam of deposited metal) is the HORIZONTAL BAND in the LOWER HALF of the image, typically between y=400 and y=800.
- ALL defect bounding boxes MUST be placed ON or immediately adjacent to the WELD BEAD — NOT above it in the bare metal zone.
- The area ABOVE y=400 is bare metal / parent plate. Do NOT place any defect box there unless the defect is explicitly a heat-affected zone defect.
- If you cannot see the defect clearly, set all values to 0 (the sentinel) so the fallback can handle it.

COORDINATE SYSTEM (relative to the FULL IMAGE):
- Coordinates must be a list of 4 integers [ymin, xmin, ymax, xmax] between 0 and 1000.
- ymin is the top edge, xmin is the left edge, ymax is the bottom edge, xmax is the right edge.

SIZING & POSITIONING RULES (CRITICAL):
- Only locate these 4 defect types: Underfill, Undercut, Excess Reinforcement, Blowhole (Porosity).
- ALL boxes MUST sit ON THE WELD BEAD (approximately y=300 to y=700). NEVER above y=250 or below y=750.
- Minimum width/height in 0-1000 scale should cover at least 40 to 80 units. NO hairline-thin boxes.
- For undercut/underfill at upper toe: ymin ~300-400, ymax ~420-580.
- For undercut/underfill at lower toe: ymin ~550-650, ymax ~670-830.
- For excess reinforcement: ymin ~350-450, ymax ~500-650.
- For blowholes: centered on the pit, minimum size 40x80.
- The box must be large enough that the defect is clearly visible through a transparent colored rectangle drawn over it.
- If you cannot see the defect clearly, set all values to 0.

Defects to locate:
{defects_json}

Return EXACTLY this JSON format, no other text:
{
  "defects": [
    {
      "defect_id": "<Exact defect_id from input>",
      "bounding_box": [0, 0, 0, 0]
    }
  ]
}
"""



def _safe_float(val) -> float | None:
    try:
        return float(val) if val is not None else None
    except (ValueError, TypeError):
        return None

def _salvage_truncated_json(raw: str) -> dict:
    """Attempt to parse Gemini output even if truncated."""
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    try:
        or_match = re.search(r'"overall_result"\s*:\s*"(\w+)"', raw)
        qs_match = re.search(r'"weld_quality_score"\s*:\s*([\d.]+)', raw)
        overall  = or_match.group(1) if or_match else "review"
        score    = float(qs_match.group(1)) if qs_match else 0.0

        defect_pattern = re.compile(r'\{\s*"defect_id"[\s\S]*?"bounding_box"\s*:\s*(?:\{[^{}]*\}|\[[^\]]*\])[\s\S]*?\}', re.DOTALL)
        raw_defects    = defect_pattern.findall(raw)
        parsed_defects = []
        for d in raw_defects:
            try:
                parsed_defects.append(json.loads(d))
            except Exception:
                pass

        logger.warning(f"Gemini response truncated — salvaged {len(parsed_defects)} defects")
        return {
            "overall_result": overall,
            "weld_quality_score": score,
            "total_weld_length_mm": None,
            "defects": parsed_defects,
            "defect_summary": {},
            "standards_compliance": [
                {"standard": "AWS D1.1", "grade": None, "compliant": False,
                 "notes": "Partial result — response truncated"},
                {"standard": "ISO 5817", "grade": None, "compliant": False,
                 "notes": "Partial result — response truncated"},
            ],
            "recommendations": ["Re-run analysis — AI response was truncated"],
            "model_notes": "WARNING: Response truncated. Defect list may be incomplete.",
        }
    except Exception as e:
        logger.error(f"Salvage failed: {e}")

    return {
        "overall_result": "review",
        "weld_quality_score": 0.0,
        "total_weld_length_mm": None,
        "defects": [],
        "defect_summary": {},
        "standards_compliance": [
            {"standard": "AWS D1.1", "grade": None, "compliant": False, "notes": "Analysis failed"},
            {"standard": "ISO 5817", "grade": None, "compliant": False, "notes": "Analysis failed"},
        ],
        "recommendations": ["Retry — AI response could not be parsed"],
        "model_notes": "ERROR: Could not parse Gemini response.",
    }


def build_rich_text_output(image_label: str, stitched_image_url: str, raw_result: dict) -> dict:
    """
    Build the rich structured text output for the /inspect API response.

    Returns a dict with:
      - image_label
      - stitched_image_url
      - overall_result / weld_quality_score
      - scale_info (what the model read from the ruler)
      - total_weld_length_mm
      - defect_breakdown: list of per-defect detail rows
      - summary_table: list of {defect_type, total_length_mm, pct_of_weld, zones}
      - text_report: human-readable multi-line string
    """
    total_mm = raw_result.get("total_weld_length_mm") or 0.0
    scale_notes = raw_result.get("scale_notes", "Scale not detected")
    defects = raw_result.get("defects", [])
    overall = raw_result.get("overall_result", "review")
    score = raw_result.get("weld_quality_score", 0.0)
    standards = raw_result.get("standards_compliance", [])
    recommendations = raw_result.get("recommendations", [])
    model_notes = raw_result.get("model_notes", "")

    # Per-defect breakdown
    defect_breakdown = []
    # Aggregate by type for the summary table
    type_agg: dict = {}

    for d in defects:
        dtype = d.get("type", "Unknown")
        length_mm = _safe_float(d.get("length_mm"))
        start_cm = _safe_float(d.get("start_cm"))
        end_cm = _safe_float(d.get("end_cm"))
        severity = d.get("severity", "medium")
        zone = d.get("weld_zone", "")
        loc = d.get("location_description", d.get("label", ""))
        count_str = d.get("estimated_count", "")

        pct = round((length_mm / total_mm) * 100, 1) if (total_mm > 0 and length_mm) else None

        defect_breakdown.append({
            "defect_id": d.get("defect_id", ""),
            "type": dtype,
            "label": d.get("label", dtype),
            "severity": severity,
            "length_mm": length_mm,
            "start_cm": start_cm,
            "end_cm": end_cm,
            "weld_zone": zone,
            "pct_of_total_weld": pct,
            "estimated_count": count_str,
            "location_description": loc,
            "description": d.get("description", ""),
            "width_mm": _safe_float(d.get("width_mm")),
            "depth_mm": _safe_float(d.get("depth_mm")),
        })

        # Aggregate
        key = dtype.strip().title()
        if key not in type_agg:
            type_agg[key] = {"total_length_mm": 0.0, "zones": set(), "count": 0, "max_severity": severity}
        if length_mm:
            type_agg[key]["total_length_mm"] += length_mm
        if zone:
            type_agg[key]["zones"].add(zone)
        type_agg[key]["count"] += 1
        # Track worst severity
        sev_order = {"low": 0, "medium": 1, "high": 2, "critical": 3}
        if sev_order.get(severity, 0) > sev_order.get(type_agg[key]["max_severity"], 0):
            type_agg[key]["max_severity"] = severity

    # Build summary table rows
    summary_table = []
    for dtype, agg in sorted(type_agg.items()):
        total_len = round(agg["total_length_mm"], 1)
        pct = round((total_len / total_mm) * 100, 1) if total_mm > 0 and total_len else None
        summary_table.append({
            "defect_type": dtype,
            "occurrences": agg["count"],
            "total_length_mm": total_len,
            "pct_of_weld_length": pct,
            "zones_affected": sorted(list(agg["zones"])),
            "max_severity": agg["max_severity"],
        })

    # Compliance
    aws_ok = next((s.get("compliant", False) for s in standards if "AWS" in s.get("standard", "")), False)
    iso_ok = next((s.get("compliant", False) for s in standards if "ISO" in s.get("standard", "")), False)

    # Build human-readable text report
    lines = []
    lines.append("=" * 70)
    lines.append(f"WELD INSPECTION REPORT — {image_label}")
    lines.append("=" * 70)
    lines.append(f"Image URL       : {stitched_image_url}")
    lines.append(f"Overall Result  : {overall.upper()}")
    lines.append(f"Quality Score   : {score}/100")
    lines.append(f"AWS D1.1        : {'COMPLIANT' if aws_ok else 'NON-COMPLIANT'}")
    lines.append(f"ISO 5817        : {'COMPLIANT' if iso_ok else 'NON-COMPLIANT'}")
    lines.append("")
    lines.append("SCALE INFORMATION")
    lines.append("-" * 40)
    lines.append(f"  {scale_notes}")
    lines.append(f"  Total weld length in image: {total_mm:.1f} mm  ({total_mm/10:.1f} cm)")
    lines.append("")

    if defect_breakdown:
        lines.append("DEFECTS FOUND")
        lines.append("-" * 40)
        for i, d in enumerate(defect_breakdown, 1):
            lines.append(f"  [{i}] {d['label']}  [{d['severity'].upper()}]")
            if d['start_cm'] is not None and d['end_cm'] is not None:
                lines.append(f"       Position  : {d['start_cm']:.1f} cm → {d['end_cm']:.1f} cm")
            if d['length_mm'] is not None:
                pct_str = f"  ({d['pct_of_total_weld']}% of weld)" if d['pct_of_total_weld'] else ""
                lines.append(f"       Length    : {d['length_mm']:.1f} mm{pct_str}")
            if d['width_mm']:
                lines.append(f"       Width     : {d['width_mm']:.1f} mm")
            if d['depth_mm']:
                lines.append(f"       Depth     : {d['depth_mm']:.1f} mm")
            if d['weld_zone']:
                lines.append(f"       Zone      : {d['weld_zone'].replace('_', ' ').title()}")
            if d['estimated_count']:
                lines.append(f"       Count     : {d['estimated_count']}")
            lines.append(f"       Location  : {d['location_description']}")
            if d['description']:
                lines.append(f"       Remarks   : {d['description']}")
            lines.append("")
    else:
        lines.append("DEFECTS FOUND: None detected")
        lines.append("")

    lines.append("MEASUREMENT SUMMARY TABLE")
    lines.append("-" * 70)
    lines.append(f"  {'Defect Type':<28} {'Count':>6} {'Length (mm)':>12} {'% of Weld':>10}  {'Max Severity'}")
    lines.append("  " + "-" * 66)
    for row in summary_table:
        pct_str = f"{row['pct_of_weld_length']:.1f}%" if row['pct_of_weld_length'] is not None else "N/A"
        lines.append(
            f"  {row['defect_type']:<28} {row['occurrences']:>6} "
            f"{row['total_length_mm']:>12.1f} {pct_str:>10}  {row['max_severity'].upper()}"
        )
    if not summary_table:
        lines.append("  No defects detected")
    lines.append("  " + "-" * 66)
    total_defect_len = sum(r["total_length_mm"] for r in summary_table)
    total_defect_pct = round((total_defect_len / total_mm) * 100, 1) if total_mm > 0 and total_defect_len else 0.0
    lines.append(f"  {'TOTAL DEFECT LENGTH':<28} {'':>6} {total_defect_len:>12.1f} {f'{total_defect_pct:.1f}%':>10}")
    lines.append(f"  {'HEALTHY WELD LENGTH':<28} {'':>6} {max(0, total_mm - total_defect_len):>12.1f} {f'{max(0.0, 100.0 - total_defect_pct):.1f}%':>10}")
    lines.append("")

    if recommendations:
        lines.append("RECOMMENDATIONS")
        lines.append("-" * 40)
        for r in recommendations:
            lines.append(f"  • {r}")
        lines.append("")

    if model_notes:
        lines.append(f"MODEL NOTES: {model_notes}")

    lines.append("=" * 70)
    text_report = "\n".join(lines)

    return {
        "image_label": image_label,
        "stitched_image_url": stitched_image_url,
        "overall_result": overall,
        "weld_quality_score": score,
        "scale_detected": raw_result.get("scale_detected", False),
        "scale_notes": scale_notes,
        "total_weld_length_mm": total_mm,
        "defect_breakdown": defect_breakdown,
        "summary_table": summary_table,
        "total_defect_length_mm": round(total_defect_len, 1),
        "healthy_weld_length_mm": round(max(0.0, total_mm - total_defect_len), 1),
        "defect_pct_of_weld": total_defect_pct,
        "standards_compliance": raw_result.get("standards_compliance", []),
        "recommendations": recommendations,
        "model_notes": model_notes,
        "text_report": text_report,
    }


def build_object_summary_table(image_outputs: list[dict]) -> dict:
    """
    Build a cross-image summary table for all images of one object_id.
    image_outputs: list of dicts returned by build_rich_text_output()
    """
    total_weld_mm = 0.0
    type_agg: dict = {}
    pass_count = fail_count = review_count = 0
    total_images = len(image_outputs)

    for img in image_outputs:
        total_weld_mm += img.get("total_weld_length_mm") or 0.0
        res = img.get("overall_result", "review")
        if res == "pass":
            pass_count += 1
        elif res == "fail":
            fail_count += 1
        else:
            review_count += 1

        for row in img.get("summary_table", []):
            dtype = row["defect_type"]
            if dtype not in type_agg:
                type_agg[dtype] = {"total_length_mm": 0.0, "occurrences": 0, "images_affected": 0, "max_severity": "low"}
            type_agg[dtype]["total_length_mm"] += row["total_length_mm"]
            type_agg[dtype]["occurrences"] += row["occurrences"]
            type_agg[dtype]["images_affected"] += 1
            sev_order = {"low": 0, "medium": 1, "high": 2, "critical": 3}
            if sev_order.get(row["max_severity"], 0) > sev_order.get(type_agg[dtype]["max_severity"], 0):
                type_agg[dtype]["max_severity"] = row["max_severity"]

    summary_rows = []
    for dtype, agg in sorted(type_agg.items()):
        pct = round((agg["total_length_mm"] / total_weld_mm) * 100, 1) if total_weld_mm > 0 else None
        summary_rows.append({
            "defect_type": dtype,
            "occurrences": agg["occurrences"],
            "images_affected": agg["images_affected"],
            "total_length_mm": round(agg["total_length_mm"], 1),
            "pct_of_total_weld": pct,
            "max_severity": agg["max_severity"],
        })

    total_defect_len = sum(r["total_length_mm"] for r in summary_rows)

    # Build text table
    lines = []
    lines.append("=" * 80)
    lines.append("OBJECT-LEVEL INSPECTION SUMMARY")
    lines.append("=" * 80)
    lines.append(f"  Total images analyzed    : {total_images}")
    lines.append(f"  Total weld length        : {total_weld_mm:.1f} mm  ({total_weld_mm/10:.1f} cm)")
    lines.append(f"  Pass / Fail / Review     : {pass_count} / {fail_count} / {review_count}")
    lines.append("")
    lines.append(f"  {'Defect Type':<28} {'Occur':>6} {'Images':>7} {'Total mm':>10} {'% of Weld':>10}  {'Max Sev'}")
    lines.append("  " + "-" * 70)
    for row in summary_rows:
        pct_str = f"{row['pct_of_total_weld']:.1f}%" if row["pct_of_total_weld"] is not None else "N/A"
        lines.append(
            f"  {row['defect_type']:<28} {row['occurrences']:>6} {row['images_affected']:>7} "
            f"{row['total_length_mm']:>10.1f} {pct_str:>10}  {row['max_severity'].upper()}"
        )
    if not summary_rows:
        lines.append("  No defects detected across all images")
    lines.append("  " + "-" * 70)
    total_pct = round((total_defect_len / total_weld_mm) * 100, 1) if total_weld_mm > 0 else 0.0
    lines.append(f"  {'TOTAL DEFECT LENGTH':<28} {'':>6} {'':>7} {total_defect_len:>10.1f} {f'{total_pct:.1f}%':>10}")
    lines.append(f"  {'HEALTHY WELD':<28} {'':>6} {'':>7} {max(0, total_weld_mm - total_defect_len):>10.1f} {f'{max(0.0, 100.0 - total_pct):.1f}%':>10}")
    lines.append("=" * 80)

    return {
        "total_images": total_images,
        "total_weld_length_mm": round(total_weld_mm, 1),
        "pass_count": pass_count,
        "fail_count": fail_count,
        "review_count": review_count,
        "summary_rows": summary_rows,
        "total_defect_length_mm": round(total_defect_len, 1),
        "healthy_weld_length_mm": round(max(0.0, total_weld_mm - total_defect_len), 1),
        "text_table": "\n".join(lines),
    }


def _estimate_bbox_from_text(defect: dict, total_weld_mm: float) -> dict | None:
    """
    Fallback: compute an approximate bounding box from the text location fields
    when Gemini's Pass 2 localization fails to return a bbox for a defect.

    Uses start_cm/end_cm for horizontal position, and location_side + weld_zone
    for vertical position within the image.

    NOTE: y-values are anchored to the WELD BEAD which sits in the lower half
    of the stitched image (approximately y=0.42 to y=0.78). The upper portion
    of the image is bare parent plate — defect boxes must NOT be placed there.
    """
    try:
        start_cm = defect.get("start_cm")
        end_cm   = defect.get("end_cm")
        loc_side = (defect.get("location_side") or "center").lower()
        weld_zone = (defect.get("weld_zone") or "bead_centre").lower()

        # --- Horizontal position from scale marks ---
        total_cm = total_weld_mm / 10.0 if total_weld_mm > 0 else 0
        if total_cm > 0 and start_cm is not None and end_cm is not None:
            x  = max(0.0, float(start_cm) / total_cm)
            x2 = min(1.0, float(end_cm) / total_cm)
            bw = max(0.06, x2 - x)
        elif loc_side in ("left",):
            x, bw = 0.02, 0.30
        elif loc_side in ("right",):
            x, bw = 0.68, 0.30
        elif loc_side in ("full-width", "full_width"):
            x, bw = 0.0, 1.0
        else:  # center
            x, bw = 0.35, 0.30

        # --- Vertical position from weld_zone ---
        # Weld bead sits in the LOWER half of the stitched image: ~y=0.42–0.78.
        # upper_toe  = top edge of the weld bead  (~y=0.42)
        # bead_centre = crown of the weld bead     (~y=0.50)
        # lower_toe  = bottom edge of the weld     (~y=0.60)
        # heat_affected = just above the weld bead (~y=0.38)
        # scattered   = spans the whole weld band  (~y=0.42)
        if "upper_toe" in weld_zone:
            y, bh = 0.40, 0.14
        elif "lower_toe" in weld_zone:
            y, bh = 0.58, 0.14
        elif "bead_centre" in weld_zone or "bead_center" in weld_zone:
            y, bh = 0.48, 0.16
        elif "heat_affected" in weld_zone:
            y, bh = 0.34, 0.14
        elif "scattered" in weld_zone:
            y, bh = 0.42, 0.28
        else:
            y, bh = 0.46, 0.16

        # Clamp to safe range
        x  = max(0.0, min(0.93, x))
        y  = max(0.0, min(0.80, y))
        bw = max(0.06, min(1.0 - x, bw))
        bh = max(0.06, min(0.85 - y, bh))

        return {"x": round(x, 4), "y": round(y, 4), "width": round(bw, 4), "height": round(bh, 4)}
    except Exception:
        return None


class GeminiService:
    def __init__(self):
        self.default_model_name = "gemini-2.5-pro"
        self.client = genai.Client(api_key=settings.GEMINI_API_KEY)
        self.few_shot_examples = self._load_few_shot_examples()

    def _load_few_shot_examples(self) -> list:
        import os
        dir_path = os.path.dirname(os.path.abspath(__file__))
        target_dir = os.path.join(dir_path, "few_shot_examples")
        examples = []
        
        metadata = [
            {
                "filename": "example_undercut.jpg",
                "label": "Example 1 (Undercut):",
                "response": {
                    "valid": True,
                    "overall_result": "fail",
                    "weld_quality_score": 75,
                    "scale_detected": True,
                    "scale_notes": "Scale bar spans 0 cm to 20 cm, total 20 cm = 200 mm",
                    "total_weld_length_mm": 200.0,
                    "defect_summary": {"Undercut": 1},
                    "defects": [
                        {
                            "defect_id": "example_defect_1",
                            "type": "Undercut",
                            "label": "Undercut at upper toe, 3-5 cm",
                            "description": "Visible undercut defect at the upper toe of the weld bead.",
                            "severity": "medium",
                            "confidence": 0.95,
                            "shape": "rectangle",
                            "estimated_count": "1",
                            "length_mm": 20.0,
                            "width_mm": 2.0,
                            "depth_mm": 0.5,
                            "start_cm": 3.0,
                            "end_cm": 5.0,
                            "location_side": "left",
                            "weld_zone": "upper_toe",
                            "bounding_box": [350, 150, 470, 250],
                            "location_description": "Upper toe of the weld bead, near the 3-5 cm mark"
                        }
                    ],
                    "standards_compliance": [
                        {"standard": "AWS D1.1", "compliant": False, "notes": "Undercut depth exceeds allowable limit"},
                        {"standard": "ISO 5817", "compliant": False, "notes": "Undercut depth exceeds allowable limit"}
                    ],
                    "recommendations": ["Grind and re-weld the undercut section at 3-5 cm"],
                    "model_notes": "Deterministic analysis of example image 1."
                }
            },
            {
                "filename": "example_porosity.jpg",
                "label": "Example 2 (Porosity / Blowhole):",
                "response": {
                    "valid": True,
                    "overall_result": "fail",
                    "weld_quality_score": 80,
                    "scale_detected": True,
                    "scale_notes": "Scale bar spans 0 cm to 20 cm, total 20 cm = 200 mm",
                    "total_weld_length_mm": 200.0,
                    "defect_summary": {"Blowhole (Porosity)": 1},
                    "defects": [
                        {
                            "defect_id": "example_defect_2",
                            "type": "Blowhole (Porosity)",
                            "label": "Surface porosity at bead center, 9.6-10.4 cm",
                            "description": "Surface porosity (blowhole) in the middle of the weld bead.",
                            "severity": "high",
                            "confidence": 0.98,
                            "shape": "circle",
                            "estimated_count": "1",
                            "length_mm": 8.0,
                            "width_mm": 8.0,
                            "depth_mm": 1.0,
                            "start_cm": 9.6,
                            "end_cm": 10.4,
                            "location_side": "center",
                            "weld_zone": "bead_centre",
                            "bounding_box": [450, 480, 530, 520],
                            "location_description": "Center of the weld bead, near the 10 cm mark"
                        }
                    ],
                    "standards_compliance": [
                        {"standard": "AWS D1.1", "compliant": False, "notes": "Surface porosity is not permitted under AWS D1.1 for visual inspection"},
                        {"standard": "ISO 5817", "compliant": False, "notes": "Surface porosity is not permitted"}
                    ],
                    "recommendations": ["Excavate the porosity defect and re-weld"],
                    "model_notes": "Deterministic analysis of example image 2."
                }
            },
            {
                "filename": "example_reinforcement.jpg",
                "label": "Example 3 (Excess Reinforcement):",
                "response": {
                    "valid": True,
                    "overall_result": "review",
                    "weld_quality_score": 85,
                    "scale_detected": True,
                    "scale_notes": "Scale bar spans 0 cm to 20 cm, total 20 cm = 200 mm",
                    "total_weld_length_mm": 200.0,
                    "defect_summary": {"Excess Reinforcement": 1},
                    "defects": [
                        {
                            "defect_id": "example_defect_3",
                            "type": "Excess Reinforcement",
                            "label": "Excess reinforcement at crown, 12-16 cm",
                            "description": "Weld reinforcement profile exceeds normal thickness limits.",
                            "severity": "low",
                            "confidence": 0.92,
                            "shape": "rectangle",
                            "estimated_count": "1",
                            "length_mm": 40.0,
                            "width_mm": 15.0,
                            "depth_mm": 3.0,
                            "start_cm": 12.0,
                            "end_cm": 16.0,
                            "location_side": "full-width",
                            "weld_zone": "bead_centre",
                            "bounding_box": [380, 600, 580, 800],
                            "location_description": "Weld crown reinforcement spanning from 12 cm to 16 cm"
                        }
                    ],
                    "standards_compliance": [
                        {"standard": "AWS D1.1", "compliant": True, "notes": "Reinforcement height is within acceptable limits"},
                        {"standard": "ISO 5817", "compliant": True, "notes": "Reinforcement height is within acceptable limits"}
                    ],
                    "recommendations": ["Monitor crown reinforcement height, no immediate rework needed"],
                    "model_notes": "Deterministic analysis of example image 3."
                }
            }
        ]
        
        for item in metadata:
            filepath = os.path.join(target_dir, item["filename"])
            if os.path.exists(filepath):
                try:
                    with open(filepath, "rb") as f:
                        img_bytes = f.read()
                    examples.append({
                        "label": item["label"],
                        "image_bytes": img_bytes,
                        "response": item["response"]
                    })
                except Exception as e:
                    logger.error(f"Error loading few-shot example {item['filename']}: {e}")
            else:
                logger.warning(f"Few-shot example image not found: {filepath}")
                
        return examples

    async def _call_gemini(
        self,
        prompt: str,
        image_bytes: bytes,
        mime_type: str,
        model_name: str | None = None,
        temperature: float = 0.0,
        response_schema: dict | None = None,
        use_few_shot: bool = False,
        few_shot_type: str = "inspection"
    ) -> str:
        """Raw Gemini call — returns stripped text with rate limit retries."""
        import asyncio
        import time
        from google.genai.errors import APIError

        model = model_name or self.default_model_name

        config = types.GenerateContentConfig(
            temperature=temperature,
            max_output_tokens=65536,
        )
        if model != "gemini-2.5-flash-image":
            config.response_mime_type = "application/json"
            if response_schema is not None:
                config.response_schema = response_schema

        contents = []
        if use_few_shot and self.few_shot_examples:
            contents.append("Below are reference examples of weld defect analysis with their correct bounding box coordinates formatted as [ymin, xmin, ymax, xmax] on a 0-1000 scale.")
            for idx, ex in enumerate(self.few_shot_examples):
                contents.append(f"Example {idx + 1} Image:")
                contents.append(types.Part.from_bytes(data=ex["image_bytes"], mime_type="image/jpeg"))
                if few_shot_type == "inspection":
                    contents.append(f"Example {idx + 1} Output JSON:\n" + json.dumps(ex["response"], indent=2))
                elif few_shot_type == "localization":
                    loc_resp = {
                        "defects": [
                            {
                                "defect_id": d["defect_id"],
                                "bounding_box": d["bounding_box"]
                            }
                            for d in ex["response"]["defects"]
                        ]
                    }
                    contents.append(f"Example {idx + 1} Output JSON:\n" + json.dumps(loc_resp, indent=2))
            contents.append("Now analyze the following target weld image:")

        contents.append(types.Part.from_bytes(data=image_bytes, mime_type=mime_type))
        contents.append(prompt)

        for attempt in range(6):
            try:
                start = time.time()
                response = await self.client.aio.models.generate_content(
                    model=model,
                    contents=contents,
                    config=config,
                )
                elapsed = round(time.time() - start, 2)
                raw = response.text.strip()

                if "{" in raw and "}" in raw:
                    start_idx = raw.find("{")
                    end_idx = raw.rfind("}") + 1
                    raw = raw[start_idx:end_idx]

                logger.info(f"Gemini responded in {elapsed}s | {len(raw)} chars")
                return raw
            except APIError as e:
                # If it's a 503 (high demand) and we are using gemini-2.5-flash, fallback to gemini-1.5-flash
                if e.code == 503 and model == "gemini-2.5-flash" and attempt >= 2:
                    logger.warning("503 High demand on gemini-2.5-flash, falling back to gemini-1.5-flash")
                    model = "gemini-1.5-flash"
                if attempt == 5:
                    logger.error(f"Gemini API error exceeded after 6 attempts: {e}")
                    raise
                wait_time = (2 ** attempt) * 5
                logger.warning(f"Gemini API Error ({e.code}). Retrying in {wait_time}s...")
                await asyncio.sleep(wait_time)
            except Exception as e:
                if attempt == 5:
                    logger.error(f"Gemini Exception exceeded after 6 attempts: {e}")
                    raise
                wait_time = (2 ** attempt) * 5
                logger.warning(f"Gemini Error {type(e).__name__}. Retrying in {wait_time}s...")
                await asyncio.sleep(wait_time)

    async def _call_gemini_unified(self, image_bytes: bytes, mime_type: str, model_name: str | None = None) -> dict:
        return await self.inspect_with_measurements(image_bytes, mime_type=mime_type, model_name=model_name)

    async def inspect_with_measurements(
        self,
        image_bytes: bytes,
        mime_type: str = "image/jpeg",
        model_name: str | None = None,
    ) -> dict:
        """
        Single structured-output pass:
          - response_schema forces valid enums/types on every field (no more
            malformed/hallucinated field shapes to salvage).
          - property_ordering forces defect_summary (counts) to be generated
            BEFORE the itemized defects array, so the model commits to "how many"
            before "where each one is" — this is the count-consistency check.
          - temperature is lowered for this call specifically since this is a
            precision/measurement task, not a creative one.
          - Low-confidence entries (model's own stated confidence < 0.5) are
            dropped rather than trusted, since it's better to omit a borderline
            defect than draw a hallucinated box.
        """
        try:
            logger.info("[inspect] Structured weld analysis started")
            raw = await self._call_gemini(
                WELD_MEASUREMENT_INSPECTION_PROMPT, image_bytes, mime_type,
                model_name=model_name, temperature=0.0,
                response_schema=WELD_INSPECTION_RESPONSE_SCHEMA,
                use_few_shot=True, few_shot_type="inspection"
            )
            data = _salvage_truncated_json(raw)

            if not data.get("valid", True):
                reason = data.get("reason", "Not a weld image")
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                    detail=f"Image rejected: {reason}",
                )

            defects = data.get("defects", [])
            logger.info(f"[inspect] Analysis complete: {len(defects)} defects identified")

            # Drop anything the model itself wasn't confident about.
            before = len(defects)
            defects = [
                d for d in defects
                if isinstance(d, dict) and float(d.get("confidence", 1.0)) >= 0.5
            ]
            if len(defects) != before:
                logger.info(f"[inspect] Dropped {before - len(defects)} low-confidence defect(s)")

            # Call Pass 2 localization for high-precision snapped bounding boxes
            if defects:
                try:
                    logger.info(f"[inspect] Running Pass 2 localization for {len(defects)} defects")
                    bboxes = await self.locate_defects(image_bytes, defects, mime_type, model_name)
                    norm_bboxes = {str(k).strip().lower(): v for k, v in bboxes.items() if k}
                    for d in defects:
                        did = d.get("defect_id")
                        norm_did = str(did).strip().lower() if did else ""
                        if norm_did in norm_bboxes:
                            d["bounding_box"] = norm_bboxes[norm_did].model_dump()
                        else:
                            # Fallback: estimate from text
                            est_bb = _estimate_bbox_from_text(d, data.get("total_weld_length_mm") or 200.0)
                            d["bounding_box"] = est_bb
                except Exception as e:
                    logger.error(f"Pass 2 localization failed, falling back to text estimation: {e}")
                    for d in defects:
                        est_bb = _estimate_bbox_from_text(d, data.get("total_weld_length_mm") or 200.0)
                        d["bounding_box"] = est_bb

            # Count-consistency check: compare defect_summary counts against what
            # was actually itemized. Mismatches are logged (signal for monitoring/
            # prompt tuning) — the itemized list is treated as source of truth
            # for what actually gets drawn.
            summary = data.get("defect_summary", {}) or {}
            actual_counts: dict = {}
            for d in defects:
                key = str(d.get("type", "")).strip().title()
                actual_counts[key] = actual_counts.get(key, 0) + 1
            for stype, scount in summary.items():
                acount = actual_counts.get(str(stype).strip().title(), 0)
                if acount and int(scount) != acount:
                    logger.warning(
                        f"[inspect] Count mismatch for '{stype}': "
                        f"summary said {scount}, itemized {acount}"
                    )

            data["defects"] = defects
            return data

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Gemini measurement inspection failed: {e}")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"AI measurement inspection failed: {str(e)}",
            )

    def _apply_weld_bead_constraints(self, defects: list, loc_data: dict) -> dict:
        import math
        weld_beads = loc_data.get("weld_beads", [])
        valid_weld_beads = []
        for wb in weld_beads:
            if isinstance(wb, list) and len(wb) >= 4:
                ymin, xmin, ymax, xmax = wb[0], wb[1], wb[2], wb[3]
                wb = {
                    "x": xmin / 1000.0,
                    "y": ymin / 1000.0,
                    "width": (xmax - xmin) / 1000.0,
                    "height": (ymax - ymin) / 1000.0,
                }
            if isinstance(wb, dict):
                wx = max(0.0, min(1.0, float(wb.get("x", 0))))
                wy = max(0.0, min(1.0, float(wb.get("y", 0))))
                ww = max(0.01, min(1.0 - wx, float(wb.get("width", 0.05))))
                wh = max(0.01, min(1.0 - wy, float(wb.get("height", 0.05))))
                if ww > 0.02 or wh > 0.02:
                    valid_weld_beads.append({"x": wx, "y": wy, "width": ww, "height": wh})

        bb_map: dict = {}
        for entry in loc_data.get("defects", []):
            did = entry.get("defect_id")
            bb  = entry.get("bounding_box")
            if did and isinstance(bb, list) and len(bb) >= 4:
                ymin, xmin, ymax, xmax = bb[0], bb[1], bb[2], bb[3]
                bb = {
                    "x": xmin / 1000.0,
                    "y": ymin / 1000.0,
                    "width": (xmax - xmin) / 1000.0,
                    "height": (ymax - ymin) / 1000.0,
                }
            if did and isinstance(bb, dict):
                x = max(0.0, min(0.99, float(bb.get("x", 0))))
                y = max(0.0, min(0.99, float(bb.get("y", 0))))
                bw = max(0.01, min(1.0 - x, float(bb.get("width", 0.05))))
                bh = max(0.01, min(1.0 - y, float(bb.get("height", 0.05))))

                if x == 0 and y == 0 and bw <= 0.01 and bh <= 0.01:
                    continue

                defect_type = ""
                for d in defects:
                    d_id = d.get("defect_id") if isinstance(d, dict) else getattr(d, "defect_id", None)
                    if d_id and str(d_id).strip().lower() == str(did).strip().lower():
                        defect_type = (d.get("type", "") if isinstance(d, dict) else getattr(d, "type", "")).lower()
                        break

                is_weld_defect = any(k in defect_type for k in [
                    "undercut", "underfill", "fusion", "penetration", "profile",
                    "reinforcement", "hump", "blowhole", "porosity", "crack",
                    "overlap", "crater", "burn", "spatter", "slag"
                ])

                if valid_weld_beads and is_weld_defect:
                    dcx = x + bw / 2
                    dcy = y + bh / 2
                    nearest_wb = min(
                        valid_weld_beads,
                        key=lambda wb: math.hypot(dcx - (wb["x"] + wb["width"]/2), dcy - (wb["y"] + wb["height"]/2))
                    )
                    wb_x_left   = nearest_wb["x"]
                    wb_x_right  = nearest_wb["x"] + nearest_wb["width"]
                    wb_y_top    = nearest_wb["y"]
                    wb_y_bottom = nearest_wb["y"] + nearest_wb["height"]
                    margin_x = margin_y = 0.05
                    x = max(wb_x_left - margin_x, min(wb_x_right + margin_x - bw, x))
                    y = max(wb_y_top - margin_y, min(wb_y_bottom + margin_y - bh, y))

                x = max(0.0, min(1.0 - bw, x))
                y = max(0.0, min(1.0 - bh, y))
                bb_map[did] = {"x": x, "y": y, "width": bw, "height": bh}

        return bb_map

    def validate_weld_image(self, image_bytes: bytes, mime_type: str = "image/jpeg") -> tuple[bool, str]:
        return True, "Validation merged into unified prompt"

    async def analyze_pair(
        self,
        stitched_bytes: bytes,
        frame_index: int,
        image_label: str,
        source_frame_a_label: str,
        source_frame_b_label: str | None,
        timestamp_a: float,
        timestamp_b: float | None,
        raw_frame_a_url: str,
        raw_frame_b_url: str | None,
        stitched_image_url: str,
        annotated_image_url: str,
        start_cm: float = 0.0,
        length_cm: float = 0.0,
        mime_type: str = "image/jpeg",
        model_name: str | None = None,
    ) -> FramePairResult:
        raw = await self._call_gemini_unified(stitched_bytes, mime_type, model_name=model_name)

        defects = []
        for d in raw.get("defects", []):
            bb = d.get("bounding_box")
            try:
                if isinstance(bb, list) and len(bb) >= 4:
                    ymin, xmin, ymax, xmax = bb[0], bb[1], bb[2], bb[3]
                    bb = {
                        "x": max(0.0, min(1.0, float(xmin) / 1000.0)),
                        "y": max(0.0, min(1.0, float(ymin) / 1000.0)),
                        "width": max(0.001, min(1.0 - float(xmin)/1000.0, float(xmax - xmin) / 1000.0)),
                        "height": max(0.001, min(1.0 - float(ymin)/1000.0, float(ymax - ymin) / 1000.0)),
                    }
                if isinstance(bb, dict):
                    bb = {
                        "x": max(0.0, min(1.0, float(bb.get("x", 0)))),
                        "y": max(0.0, min(1.0, float(bb.get("y", 0)))),
                        "width": max(0.01, min(1.0, float(bb.get("width", 0.1)))),
                        "height": max(0.01, min(1.0, float(bb.get("height", 0.1)))),
                    }
                else:
                    bb = None

                sev_raw = str(d.get("severity", "medium")).lower()
                sev = "critical" if "crit" in sev_raw else ("high" if "high" in sev_raw else ("low" if "low" in sev_raw else "medium"))

                length_mm = _safe_float(d.get("length_mm"))
                if (length_mm is None or length_mm == 0.0) and length_cm > 0.0 and bb:
                    length_mm = round(bb["width"] * length_cm * 10.0, 1)

                est_count = str(d.get("estimated_count")) if d.get("estimated_count") is not None else None
                shape = d.get("shape")
                if shape not in ("rectangle", "square", "circle", "oval"):
                    shape = None
                try:
                    confidence = float(d.get("confidence", 1.0))
                except (TypeError, ValueError):
                    confidence = 1.0
                confidence = max(0.0, min(1.0, confidence))

                defects.append(Defect(
                    defect_id=str(d.get("defect_id", str(uuid.uuid4())[:8])),
                    type=str(d.get("type", "Unknown")),
                    label=d.get("label"),
                    severity=DefectSeverity(sev),
                    description=str(d.get("description", "")),
                    confidence=confidence,
                    shape=shape,
                    bounding_box=BoundingBox(**bb) if bb else None,
                    length_mm=length_mm,
                    depth_mm=_safe_float(d.get("depth_mm")),
                    width_mm=_safe_float(d.get("width_mm")),
                    count=d.get("count"),
                    estimated_count=est_count,
                    position=d.get("position"),
                    standards_reference=d.get("standards_reference"),
                    recommendation=d.get("recommendation"),
                ))
            except Exception as e:
                logger.warning(f"Skipping malformed defect entry: {e}")
                continue

        standards = []
        for s in raw.get("standards_compliance", []):
            try:
                standards.append(WeldingStandardsCompliance(
                    standard=s.get("standard", ""),
                    grade=s.get("grade"),
                    compliant=bool(s.get("compliant", False)),
                    notes=s.get("notes"),
                ))
            except Exception as e:
                logger.warning(f"Skipping malformed compliance entry: {e}")

        raw_overall = str(raw.get("overall_result", "review")).lower()
        overall = "pass" if "pass" in raw_overall else ("fail" if "fail" in raw_overall else "review")

        return FramePairResult(
            frame_index=frame_index,
            image_label=image_label,
            source_frame_a_label=source_frame_a_label,
            source_frame_b_label=source_frame_b_label,
            timestamp_a_seconds=timestamp_a,
            timestamp_b_seconds=timestamp_b,
            raw_frame_a_url=raw_frame_a_url,
            raw_frame_b_url=raw_frame_b_url,
            stitched_image_url=stitched_image_url,
            annotated_image_url=annotated_image_url,
            overall_result=OverallResult(overall),
            weld_quality_score=float(raw.get("weld_quality_score", 0.0)),
            defects=defects,
            defect_summary=raw.get("defect_summary", {}),
            standards_compliance=standards,
            recommendations=raw.get("recommendations", []),
            model_notes=raw.get("model_notes"),
        )

    async def custom_image_inspection(self, image_bytes: bytes, mime_type: str = "image/jpeg") -> str:
        prompt = "check the image and find welding defects like undercut, underfill, blowholes, excess reinforcement and spatters and every possible defect, mark all defects and create one image with all defects marked down precisely"
        import asyncio
        for attempt in range(3):
            try:
                response = await self.client.aio.models.generate_content(
                    model=self.default_model_name,
                    contents=[prompt, types.Part.from_bytes(data=image_bytes, mime_type=mime_type)],
                    config=types.GenerateContentConfig(temperature=0.0),
                )
                return response.text
            except Exception as e:
                if attempt == 2:
                    raise
                await asyncio.sleep(2)

    async def _localize_defects_with_crop(self, image_bytes: bytes, defects: list, mime_type: str, model_name: str | None) -> dict:
        logger.info("[inspect] Localizing weld beads")
        wb_raw = await self._call_gemini(
            WELD_BEAD_LOCALIZATION_PROMPT, image_bytes, mime_type, model_name=model_name,
            temperature=0.0, response_schema=WELD_BEAD_LOCALIZATION_RESPONSE_SCHEMA
        )
        wb_data = _salvage_truncated_json(wb_raw)
        weld_beads = wb_data.get("weld_beads", [])

        valid_weld_beads = []
        for wb in weld_beads:
            if isinstance(wb, list) and len(wb) >= 4:
                ymin, xmin, ymax, xmax = wb[0], wb[1], wb[2], wb[3]
                wb = {
                    "x": xmin / 1000.0,
                    "y": ymin / 1000.0,
                    "width": (xmax - xmin) / 1000.0,
                    "height": (ymax - ymin) / 1000.0,
                }
            if isinstance(wb, dict):
                wx = max(0.0, min(1.0, float(wb.get("x", 0))))
                wy = max(0.0, min(1.0, float(wb.get("y", 0))))
                ww = max(0.01, min(1.0 - wx, float(wb.get("width", 0.05))))
                wh = max(0.01, min(1.0 - wy, float(wb.get("height", 0.05))))
                if ww > 0.02 or wh > 0.02:
                    valid_weld_beads.append({"x": wx, "y": wy, "width": ww, "height": wh})

        if not valid_weld_beads:
            logger.warning("[inspect] No weld beads found, falling back to full image")
            valid_weld_beads = [{"x": 0.0, "y": 0.0, "width": 1.0, "height": 1.0}]

        from PIL import Image
        import io

        img = Image.open(io.BytesIO(image_bytes))
        img_w, img_h = img.size

        loc_input = []
        for d in defects:
            if isinstance(d, dict):
                loc_input.append({
                    "defect_id": d.get("defect_id"),
                    "type": d.get("type"),
                    "label": d.get("label", ""),
                    "location_description": d.get("location_description", ""),
                })
            else:
                loc_input.append({
                    "defect_id": d.defect_id,
                    "type": d.type,
                    "label": d.label,
                    "location_description": getattr(d, "position", None) or d.description,
                })

        merged_defects = {}

        for wb in valid_weld_beads:
            pad_x = 0.1
            pad_y = 0.2
            ex = max(0.0, wb["x"] - wb["width"] * pad_x)
            ey = max(0.0, wb["y"] - wb["height"] * pad_y)
            ew = min(1.0 - ex, wb["width"] * (1.0 + 2 * pad_x))
            eh = min(1.0 - ey, wb["height"] * (1.0 + 2 * pad_y))

            crop_x1 = int(ex * img_w)
            crop_y1 = int(ey * img_h)
            crop_x2 = int((ex + ew) * img_w)
            crop_y2 = int((ey + eh) * img_h)

            if crop_x2 <= crop_x1 or crop_y2 <= crop_y1:
                continue

            crop_img = img.crop((crop_x1, crop_y1, crop_x2, crop_y2))
            crop_bytes_io = io.BytesIO()
            crop_img.save(crop_bytes_io, format="JPEG", quality=98, subsampling=0)
            crop_bytes = crop_bytes_io.getvalue()

            loc_prompt = WELD_DEFECT_LOCALIZATION_PROMPT.replace(
                "{defects_json}", json.dumps(loc_input, indent=2)
            )

            def_raw = await self._call_gemini(
                loc_prompt, crop_bytes, mime_type, model_name=model_name,
                temperature=0.0, response_schema=WELD_DEFECT_LOCALIZATION_RESPONSE_SCHEMA,
                use_few_shot=True, few_shot_type="localization"
            )
            def_data = _salvage_truncated_json(def_raw)

            for entry in def_data.get("defects", []):
                did = entry.get("defect_id")
                bb = entry.get("bounding_box")
                if did and isinstance(bb, list) and len(bb) >= 4:
                    ymin, xmin, ymax, xmax = bb[0], bb[1], bb[2], bb[3]
                    cx = float(xmin) / 1000.0
                    cy = float(ymin) / 1000.0
                    cw = float(xmax - xmin) / 1000.0
                    ch = float(ymax - ymin) / 1000.0

                    if cw > 0.01 and ch > 0.01:
                        full_x = ex + cx * ew
                        full_y = ey + cy * eh
                        full_w = cw * ew
                        full_h = ch * eh

                        if did not in merged_defects:
                            merged_defects[did] = {
                                "x": max(0.0, min(1.0, full_x)),
                                "y": max(0.0, min(1.0, full_y)),
                                "width": max(0.01, min(1.0 - full_x, full_w)),
                                "height": max(0.01, min(1.0 - full_y, full_h)),
                            }

        return {
            "weld_beads": valid_weld_beads,
            "defects": [
                {"defect_id": did, "bounding_box": bb}
                for did, bb in merged_defects.items()
            ],
        }

    async def locate_defects(self, image_bytes: bytes, defects: list, mime_type: str = "image/jpeg", model_name: str | None = None) -> dict:
        if not defects:
            return {}
        loc_data = await self._localize_defects_with_crop(image_bytes, defects, mime_type, model_name)
        bb_map = self._apply_weld_bead_constraints(defects, loc_data)
        bboxes = {}
        for did, bb in bb_map.items():
            bboxes[did] = BoundingBox(x=bb["x"], y=bb["y"], width=bb["width"], height=bb["height"])
        return bboxes


gemini_service = GeminiService()