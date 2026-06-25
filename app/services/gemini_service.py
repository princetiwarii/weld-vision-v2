"""
gemini_service.py
=================
Wraps Google Gemini Vision API for weld defect analysis.

Uses google.genai (new SDK) — NOT the deprecated google.generativeai.

Each call receives a **stitched image** (two consecutive frames joined
side-by-side). Includes:
  - Weld image validation (reject non-weld images)
  - Rich, precise bounding box instructions
  - Two-pass crop localization for accurate bounding boxes
  - Production-grade error handling and JSON salvage
"""
from google import genai
from google.genai import types
import json
import re
import uuid
import io
import time
import asyncio
from fastapi import HTTPException, status
from app.core.config import settings
from app.schemas.inspection import (
    Defect, BoundingBox, WeldingStandardsCompliance,
    DefectSeverity, OverallResult, FramePairResult,
)
from loguru import logger

# ---------------------------------------------------------------------------
# Client (new SDK)
# ---------------------------------------------------------------------------
_client = genai.Client(api_key=settings.GEMINI_API_KEY)

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------
WELD_MEASUREMENT_INSPECTION_PROMPT = """\
Act as an expert Certified Welding Inspector (CWI).

I have provided an image of a weld. Your task is to:
1. Locate the physical scale/ruler in the image (usually at the bottom).
2. Determine the physical dimensions of the weld based on that scale.
3. Inspect the entire weld for all types of defects (blowholes, porosity,
   undercut, underfill, excess reinforcement, spatter, lack of fusion, cracks, etc.).
4. For each defect found, calculate its physical measurements (length_mm,
   width_mm, depth_mm if applicable) based strictly on the scale you read.
5. Provide precise bounding box coordinates for each defect, tightly clamped
   to the visible defect boundary (0.0–1.0, x=0 is left, y=0 is top).

CRITICAL INSTRUCTIONS TO PREVENT TRUNCATION:
- Do NOT list dozens of tiny individual defects.
- Group widespread/clustered defects (spatter, porosity) into a SINGLE large
  bounding box encompassing the area.
- Limit output to a MAXIMUM of 15 major defect zones per image.

Return EXACTLY in this JSON format:
{
  "valid": true,
  "overall_result": "pass"|"fail"|"review",
  "weld_quality_score": 90,
  "scale_detected": true,
  "scale_notes": "<Brief explanation of scale, e.g. '0 to 80 cm scale found at bottom'>",
  "defect_summary": {"Spatter": 2, "Undercut": 1},
  "defects": [
    {
      "defect_id": "Seq 1",
      "type": "<Defect Type>",
      "label": "<Descriptive Label>",
      "description": "<Remarks>",
      "severity": "low"|"medium"|"high"|"critical",
      "estimated_count": "<Quantity if applicable>",
      "length_mm": 10.5,
      "width_mm": 2.1,
      "depth_mm": 0.0,
      "location_description": "<Precise textual description>",
      "bounding_box": {"x": 0.0, "y": 0.0, "width": 0.0, "height": 0.0}
    }
  ],
  "standards_compliance": [
    {"standard": "AWS D1.1", "compliant": false, "notes": "<Reason>"}
  ],
  "recommendations": ["<Actionable recommendation>"],
  "model_notes": "<Any additional notes>"
}
"""

WELD_BEAD_LOCALIZATION_PROMPT = """\
You are a computer vision expert. Locate all visible weld beads in the image.
Return ONLY this JSON:
{
  "weld_beads": [
    {"x": 0.0, "y": 0.0, "width": 1.0, "height": 0.3}
  ]
}
Where x, y, width, height are normalized 0.0–1.0 coordinates.
"""

WELD_DEFECT_LOCALIZATION_PROMPT = """\
You are an expert computer vision system. I have already identified a list of
weld defects in the provided IMAGE CROP. Your ONLY job is to locate these
specific defects and provide their precise bounding box coordinates within
this crop.

The bounding box format MUST be an object with normalized coordinates
0.0–1.0, where: x = left edge, y = top edge, width = box width, height = box height.

Focus entirely on tight precision around the defect.
DO NOT identify new defects. ONLY locate the ones provided.

Defects to locate:
{defects_json}

Return EXACTLY this JSON:
{
  "defects": [
    {
      "defect_id": "<Exact defect_id from input>",
      "bounding_box": {"x": 0.0, "y": 0.0, "width": 0.0, "height": 0.0}
    }
  ]
}
"""


# ---------------------------------------------------------------------------
# JSON helpers
# ---------------------------------------------------------------------------
def _salvage_truncated_json(raw: str) -> dict:
    """Attempt to parse Gemini output even if truncated."""
    # Strip markdown fences
    raw = raw.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```[a-z]*\n?", "", raw)
        raw = re.sub(r"\n?```$", "", raw)

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    # Extract the outermost JSON object
    try:
        start = raw.index("{")
        end   = raw.rindex("}") + 1
        return json.loads(raw[start:end])
    except (ValueError, json.JSONDecodeError):
        pass

    try:
        or_match = re.search(r'"overall_result"\s*:\s*"(\w+)"', raw)
        qs_match = re.search(r'"weld_quality_score"\s*:\s*([\d.]+)', raw)
        overall  = or_match.group(1) if or_match else "review"
        score    = float(qs_match.group(1)) if qs_match else 0.0

        defect_pattern = re.compile(
            r'\{\s*"defect_id"[\s\S]*?"bounding_box"\s*:\s*\{[^{}]*\}[\s\S]*?\}',
            re.DOTALL
        )
        parsed_defects = []
        for d in defect_pattern.findall(raw):
            try:
                parsed_defects.append(json.loads(d))
            except Exception:
                pass

        logger.warning(f"Gemini response truncated — salvaged {len(parsed_defects)} defects")
        return {
            "overall_result": overall,
            "weld_quality_score": score,
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
        "defects": [],
        "defect_summary": {},
        "standards_compliance": [
            {"standard": "AWS D1.1", "grade": None, "compliant": False, "notes": "Analysis failed"},
            {"standard": "ISO 5817", "grade": None, "compliant": False, "notes": "Analysis failed"},
        ],
        "recommendations": ["Retry — AI response could not be parsed"],
        "model_notes": "ERROR: Could not parse Gemini response.",
    }


# ---------------------------------------------------------------------------
# Output formatters (used by sessions.py)
# ---------------------------------------------------------------------------
def build_rich_text_output(image_label: str, stitched_image_url: str, raw_result: dict) -> dict:
    """Build a structured per-image output dict for the inspect endpoint response."""
    defects_out = []
    for d in raw_result.get("defects", []):
        bb = d.get("bounding_box")
        defects_out.append({
            "defect_id":           d.get("defect_id"),
            "type":                d.get("type"),
            "label":               d.get("label"),
            "severity":            d.get("severity"),
            "description":         d.get("description"),
            "estimated_count":     d.get("estimated_count"),
            "length_mm":           d.get("length_mm"),
            "width_mm":            d.get("width_mm"),
            "depth_mm":            d.get("depth_mm"),
            "location_description":d.get("location_description"),
            "bounding_box":        bb,
        })

    return {
        "image_label":        image_label,
        "stitched_image_url": stitched_image_url,
        "overall_result":     raw_result.get("overall_result", "review"),
        "weld_quality_score": raw_result.get("weld_quality_score", 0.0),
        "scale_detected":     raw_result.get("scale_detected", False),
        "scale_notes":        raw_result.get("scale_notes"),
        "defect_summary":     raw_result.get("defect_summary", {}),
        "defects":            defects_out,
        "standards_compliance": raw_result.get("standards_compliance", []),
        "recommendations":    raw_result.get("recommendations", []),
        "model_notes":        raw_result.get("model_notes"),
    }


def build_object_summary_table(frames_out: list) -> dict:
    """Aggregate per-frame results into a cross-image summary."""
    all_defects = []
    for f in frames_out:
        for d in f.get("defects", []):
            all_defects.append({**d, "image_label": f.get("image_label")})

    by_type: dict[str, list] = {}
    for d in all_defects:
        t = d.get("type", "Unknown")
        by_type.setdefault(t, []).append(d)

    summary_rows = []
    for t, ds in by_type.items():
        summary_rows.append({
            "type":           t,
            "count":          len(ds),
            "avg_length_mm":  round(
                sum(x.get("length_mm") or 0 for x in ds) / len(ds), 1
            ) if ds else None,
            "max_severity":   max(
                (x.get("severity") or "low") for x in ds
            ),
            "images_affected": list({x.get("image_label") for x in ds}),
        })

    overall_scores = [f.get("weld_quality_score", 0) for f in frames_out if f.get("weld_quality_score")]
    return {
        "total_defects":      len(all_defects),
        "defect_type_summary": summary_rows,
        "avg_quality_score":  round(sum(overall_scores) / len(overall_scores), 1) if overall_scores else 0.0,
        "images_analyzed":    len(frames_out),
    }


# ---------------------------------------------------------------------------
# GeminiService
# ---------------------------------------------------------------------------
class GeminiService:
    def __init__(self):
        self.client = _client
        self.default_model_name = "gemini-2.5-flash"

    async def _call_gemini(
        self,
        prompt: str,
        image_bytes: bytes,
        mime_type: str,
        model_name: str | None = None,
    ) -> str:
        """Raw Gemini call — returns stripped JSON text with rate-limit retries."""
        model = model_name or self.default_model_name

        for attempt in range(4):
            try:
                t0 = time.time()
                response = await self.client.aio.models.generate_content(
                    model=model,
                    contents=[
                        prompt,
                        types.Part.from_bytes(data=image_bytes, mime_type=mime_type),
                    ],
                    config=types.GenerateContentConfig(
                        temperature=0.6,
                        max_output_tokens=8192,
                        response_mime_type="application/json",
                    ),
                )
                elapsed = round(time.time() - t0, 2)
                raw = response.text.strip() if response.text else ""

                # Extract outermost JSON block
                if "{" in raw and "}" in raw:
                    raw = raw[raw.index("{") : raw.rindex("}") + 1]

                logger.info(f"Gemini responded in {elapsed}s | {len(raw)} chars")
                return raw

            except Exception as e:
                err_name = type(e).__name__
                if attempt == 3:
                    logger.error(f"Gemini API error after 4 attempts ({err_name}): {e}")
                    raise
                wait = (2 ** attempt) * 5
                logger.warning(f"Gemini {err_name} — retrying in {wait}s (attempt {attempt+1}/4)")
                await asyncio.sleep(wait)

    async def inspect_with_measurements(
        self,
        image_bytes: bytes,
        mime_type: str = "image/jpeg",
        model_name: str | None = None,
    ) -> dict:
        """Pass 1: full weld inspection with scale-aware physical measurements."""
        try:
            raw  = await self._call_gemini(
                WELD_MEASUREMENT_INSPECTION_PROMPT, image_bytes, mime_type, model_name
            )
            data = _salvage_truncated_json(raw)

            if not data.get("valid", True):
                reason = data.get("reason", "Not a weld image")
                logger.warning(f"Non-weld image rejected: {reason}")
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                    detail=f"Image rejected: {reason}",
                )
            return data

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Gemini measurement inspection failed: {e}")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"AI measurement inspection failed: {str(e)}",
            )

    async def _localize_defects_with_crop(
        self,
        image_bytes: bytes,
        defects: list,
        mime_type: str,
        model_name: str | None,
    ) -> dict:
        """Pass 2: Crop each weld bead region and ask Gemini to localize defects precisely."""
        logger.info("[inspect] Localizing weld beads")
        wb_raw  = await self._call_gemini(WELD_BEAD_LOCALIZATION_PROMPT, image_bytes, mime_type, model_name=model_name)
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
        img = Image.open(io.BytesIO(image_bytes))
        img_w, img_h = img.size

        loc_input = []
        for d in defects:
            if isinstance(d, dict):
                loc_input.append({
                    "defect_id":           d.get("defect_id"),
                    "type":                d.get("type"),
                    "label":               d.get("label", ""),
                    "location_description":d.get("location_description", ""),
                })
            else:
                loc_input.append({
                    "defect_id":           d.defect_id,
                    "type":                d.type,
                    "label":               d.label,
                    "location_description":getattr(d, "position", None) or d.description,
                })

        merged_defects: dict = {}

        for wb in valid_weld_beads:
            pad_x = 0.1
            pad_y = 0.2
            ex = max(0.0, wb["x"] - wb["width"]  * pad_x)
            ey = max(0.0, wb["y"] - wb["height"] * pad_y)
            ew = min(1.0 - ex, wb["width"]  * (1.0 + 2 * pad_x))
            eh = min(1.0 - ey, wb["height"] * (1.0 + 2 * pad_y))

            crop_x1 = int(ex * img_w)
            crop_y1 = int(ey * img_h)
            crop_x2 = int((ex + ew) * img_w)
            crop_y2 = int((ey + eh) * img_h)

            if crop_x2 <= crop_x1 or crop_y2 <= crop_y1:
                continue

            crop_img = img.crop((crop_x1, crop_y1, crop_x2, crop_y2))
            buf = io.BytesIO()
            crop_img.save(buf, format="JPEG")
            crop_bytes = buf.getvalue()

            loc_prompt = WELD_DEFECT_LOCALIZATION_PROMPT.replace(
                "{defects_json}", json.dumps(loc_input, indent=2)
            )

            def_raw  = await self._call_gemini(loc_prompt, crop_bytes, mime_type, model_name=model_name)
            def_data = _salvage_truncated_json(def_raw)

            for entry in def_data.get("defects", []):
                did = entry.get("defect_id")
                bb  = entry.get("bounding_box")
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
                                "x":      max(0.0, min(1.0, full_x)),
                                "y":      max(0.0, min(1.0, full_y)),
                                "width":  max(0.01, min(1.0 - full_x, full_w)),
                                "height": max(0.01, min(1.0 - full_y, full_h)),
                            }

        return {
            "weld_beads": valid_weld_beads,
            "defects": [
                {"defect_id": did, "bounding_box": bb}
                for did, bb in merged_defects.items()
            ],
        }

    def _apply_weld_bead_constraints(self, defects: list, loc_data: dict) -> dict:
        """Map localized bounding boxes back to defect objects."""
        bb_map = {e["defect_id"]: e["bounding_box"] for e in loc_data.get("defects", [])}
        result = {}
        for d in defects:
            did = d.defect_id if hasattr(d, "defect_id") else d.get("defect_id")
            if did and did in bb_map:
                result[did] = bb_map[did]
        return result

    async def locate_defects(
        self,
        image_bytes: bytes,
        defects: list,
        mime_type: str = "image/jpeg",
        model_name: str | None = None,
    ) -> dict:
        """Public entry point for Pass 2 localization. Returns {defect_id: BoundingBox}."""
        if not defects:
            return {}
        loc_data = await self._localize_defects_with_crop(image_bytes, defects, mime_type, model_name)
        bb_map   = self._apply_weld_bead_constraints(defects, loc_data)
        bboxes   = {}
        for did, bb in bb_map.items():
            bboxes[did] = BoundingBox(
                x=bb["x"], y=bb["y"], width=bb["width"], height=bb["height"]
            )
        return bboxes

    async def custom_image_inspection(
        self, image_bytes: bytes, mime_type: str = "image/jpeg"
    ) -> str:
        prompt = (
            "Check the image and find welding defects like undercut, underfill, "
            "blowholes, excess reinforcement and spatters and every possible defect. "
            "Mark all defects and create one image with all defects marked down precisely."
        )
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


gemini_service = GeminiService()
