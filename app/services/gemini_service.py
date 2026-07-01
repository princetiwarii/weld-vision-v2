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
Your task is to locate the primary weld bead(s) / weld joint line(s) in the image.
Provide a list of bounding boxes, where each box tightly encloses a weld bead segment.
Ignore the background. Focus only on the actual weld metal.

The image coordinate system:
- x=0.0 is the LEFT edge, x=1.0 is the RIGHT edge
- y=0.0 is the TOP edge, y=1.0 is the BOTTOM edge
- bounding_box = {"x": left_edge, "y": top_edge, "width": box_width, "height": box_height}
- All values must be between 0.0 and 1.0

Return EXACTLY in this JSON format with no other text:
{
  "weld_beads": [
    {"x": 0.0, "y": 0.0, "width": 0.0, "height": 0.0}
  ]
}
"""

WELD_DEFECT_LOCALIZATION_PROMPT = """\
You are an expert computer vision system analyzing a cropped image of a weld bead.
I have already identified a list of weld defects.
Your task is to locate each specific defect WITHIN THIS CROPPED IMAGE and provide its precise bounding box coordinates.

The image coordinate system:
- x=0.0 is the LEFT edge, x=1.0 is the RIGHT edge
- y=0.0 is the TOP edge, y=1.0 is the BOTTOM edge
- bounding_box = {"x": left_edge, "y": top_edge, "width": box_width, "height": box_height}
- All values must be between 0.0 and 1.0
- The boxes must be TIGHT around the actual visible defect pixels.

IMPORTANT: 
- If a defect is NOT visible in this specific crop, set its bounding_box values all to 0.0.
- Defects like undercut and underfill are usually at the edges (toes) of the weld.
- Spatter may be scattered around.

Defects to locate:
{defects_json}

Return EXACTLY in this JSON format with no other text:
{
  "defects": [
    {
      "defect_id": "<Exact defect_id from input>",
      "bounding_box": {"x": 0.0, "y": 0.0, "width": 0.0, "height": 0.0}
    }
  ]
}
"""

WELD_MEASUREMENT_INSPECTION_PROMPT = """\
Act as an expert Certified Welding Inspector (CWI).

I have provided a stitched weld image. A physical scale/ruler is visible at the bottom of the image.

CRITICAL INSTRUCTION: You MUST methodically scan the image in three passes:
1. First, scan the entire length of the HORIZONTAL weld bead(s).
2. Second, scan the entire length of the VERTICAL weld bead(s).
3. Third, closely inspect the intersection (T-joint) where they meet.

Your task:
1. READ the scale bar carefully. Determine:
   - What is the total length of weld visible in this image (in mm)?
   - What is the cm/mm span of the scale ruler visible?
   Record this as "total_weld_length_mm" and "scale_notes".

2. You must prioritize and explicitly look for the following critical defects across ALL areas:
   1. Underfill
   2. Undercut
   3. Blowholes (Porosity)
   4. Excess Reinforcement
   Also note other defect types if present (spatter, lack of fusion, cracks, overlap, crater, slag, etc.).

3. For EACH defect found:
   - Measure its LENGTH along the weld axis using the scale bar as reference.
   - State exactly WHERE it is: left/right/centre of the image, upper/lower toe,
     which cm mark it starts and ends at (e.g. "from 45 cm to 50 cm").
   - CRITICAL RULE: If a defect like undercut or underfill appears in multiple separate locations, you MUST create a SEPARATE defect entry for EACH distinct location/segment. Do not just report the first one you see.
   - Group widespread clustered defects (e.g., many tiny spatter dots in the same area) into ONE zone entry.

RULES:
- Maximum 15 defect entries.
- Do NOT include bounding_box in this response.
- All measurements MUST be physically consistent with the scale you read.
- length_mm: the physical length of the defect along the weld axis.
- start_cm / end_cm: where on the ruler this defect begins and ends.
- location_side: "left" | "center" | "right" | "full-width"
- weld_zone: "upper_toe" | "bead_centre" | "lower_toe" | "heat_affected_zone" | "scattered"

Return EXACTLY this JSON, no other text:
{
  "valid": true,
  "overall_result": "pass"|"fail"|"review",
  "weld_quality_score": 85,
  "scale_detected": true,
  "scale_notes": "<e.g. Scale bar spans 40 cm to 65 cm, total 25 cm = 250 mm visible>",
  "total_weld_length_mm": 250.0,
  "defect_summary": {"Spatter": 1, "Undercut": 2},
  "defects": [
    {
      "defect_id": "Seq 1",
      "type": "<Defect Type>",
      "label": "<Short label e.g. 'Undercut at upper toe, 45–48 cm'>",
      "description": "<Detailed remarks>",
      "severity": "low"|"medium"|"high"|"critical",
      "estimated_count": "<e.g. 3 pits>",
      "length_mm": 30.0,
      "width_mm": 2.1,
      "depth_mm": 0.0,
      "start_cm": 45.0,
      "end_cm": 48.0,
      "location_side": "left",
      "weld_zone": "upper_toe",
      "location_description": "<Very precise: references scale marks, left/right, upper/lower toe>"
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

        defect_pattern = re.compile(r'\{\s*"defect_id"[\s\S]*?"bounding_box"\s*:\s*\{[^{}]*\}[\s\S]*?\}', re.DOTALL)
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


def _safe_float(val) -> float | None:
    if val is None:
        return None
    try:
        f = float(val)
        return None if str(val).lower() in ("n/a", "null", "none") else f
    except (ValueError, TypeError):
        return None


class GeminiService:
    def __init__(self):
        self.default_model_name = "gemini-2.5-flash"
        self.client = genai.Client(api_key=settings.GEMINI_API_KEY)

    async def _call_gemini(self, prompt: str, image_bytes: bytes, mime_type: str, model_name: str | None = None) -> str:
        """Raw Gemini call — returns stripped text with rate limit retries."""
        import asyncio
        import time
        from google.genai.errors import APIError

        model = model_name or self.default_model_name

        config = types.GenerateContentConfig(
            temperature=0.4,
            max_output_tokens=65536,
        )
        if model != "gemini-2.5-flash-image":
            config.response_mime_type = "application/json"

        for attempt in range(6):
            try:
                start = time.time()
                response = await self.client.aio.models.generate_content(
                    model=model,
                    contents=[
                        prompt,
                        types.Part.from_bytes(data=image_bytes, mime_type=mime_type)
                    ],
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
        try:
            raw = await self._call_gemini(WELD_INSPECTION_PROMPT, image_bytes, mime_type, model_name=model_name)
            data = _salvage_truncated_json(raw)

            if not data.get("valid", True):
                reason = data.get("reason", "Not a weld image")
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                    detail=f"Image rejected: {reason}",
                )
            return data
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Gemini API call failed: {e}")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"AI processing failed: {str(e)}",
            )

    async def inspect_with_measurements(
        self,
        image_bytes: bytes,
        mime_type: str = "image/jpeg",
        model_name: str | None = None,
    ) -> dict:
        """
        Two-pass inspection:
        Pass 1 — Inspect weld + read scale + calculate measurements (no bounding boxes).
        Pass 2/3 — Localize each defect with tight bounding boxes via cropping.
        Returns the raw_result dict (including defects with bounding_box merged in).
        """
        try:
            logger.info("[inspect] Pass 1: weld analysis + measurement started")
            raw = await self._call_gemini(
                WELD_MEASUREMENT_INSPECTION_PROMPT, image_bytes, mime_type, model_name=model_name
            )
            data = _salvage_truncated_json(raw)

            if not data.get("valid", True):
                reason = data.get("reason", "Not a weld image")
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                    detail=f"Image rejected: {reason}",
                )

            defects = data.get("defects", [])
            logger.info(f"[inspect] Pass 1 complete: {len(defects)} defects identified")

            if defects:
                loc_data = await self._localize_defects_with_crop(image_bytes, defects, mime_type, model_name)
                bb_map = self._apply_weld_bead_constraints(defects, loc_data)

                for d in defects:
                    did = d.get("defect_id")
                    if did in bb_map:
                        d["bounding_box"] = bb_map[did]
                    else:
                        d["bounding_box"] = None

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
                    if d_id == did:
                        defect_type = (d.get("type", "") if isinstance(d, dict) else getattr(d, "type", "")).lower()
                        break

                is_weld_defect = any(k in defect_type for k in [
                    "undercut", "underfill", "fusion", "penetration", "profile",
                    "reinforcement", "hump", "blowhole", "porosity", "crack",
                    "overlap", "crater", "burn"
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
                    bb = {"x": bb[0], "y": bb[1], "width": bb[2], "height": bb[3]}
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

                if length_cm > 0.0 and bb:
                    length_mm = round(bb["width"] * length_cm * 10.0, 1)
                else:
                    length_mm = _safe_float(d.get("length_mm"))

                est_count = str(d.get("estimated_count")) if d.get("estimated_count") is not None else None

                defects.append(Defect(
                    defect_id=str(d.get("defect_id", str(uuid.uuid4())[:8])),
                    type=str(d.get("type", "Unknown")),
                    label=d.get("label"),
                    severity=DefectSeverity(sev),
                    description=str(d.get("description", "")),
                    confidence=1.0,
                    bounding_box=None,
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
                    config=types.GenerateContentConfig(temperature=0.4),
                )
                return response.text
            except Exception as e:
                if attempt == 2:
                    raise
                await asyncio.sleep(2)

    async def _localize_defects_with_crop(self, image_bytes: bytes, defects: list, mime_type: str, model_name: str | None) -> dict:
        logger.info("[inspect] Localizing weld beads")
        wb_raw = await self._call_gemini(WELD_BEAD_LOCALIZATION_PROMPT, image_bytes, mime_type, model_name=model_name)
        wb_data = _salvage_truncated_json(wb_raw)
        weld_beads = wb_data.get("weld_beads", [])

        valid_weld_beads = []
        for wb in weld_beads:
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
            crop_img.save(crop_bytes_io, format="JPEG")
            crop_bytes = crop_bytes_io.getvalue()

            loc_prompt = WELD_DEFECT_LOCALIZATION_PROMPT.replace(
                "{defects_json}", json.dumps(loc_input, indent=2)
            )

            def_raw = await self._call_gemini(loc_prompt, crop_bytes, mime_type, model_name=model_name)
            def_data = _salvage_truncated_json(def_raw)

            for entry in def_data.get("defects", []):
                did = entry.get("defect_id")
                bb = entry.get("bounding_box")
                if did and isinstance(bb, dict):
                    cx = float(bb.get("x", 0))
                    cy = float(bb.get("y", 0))
                    cw = float(bb.get("width", 0))
                    ch = float(bb.get("height", 0))

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
