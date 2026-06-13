"""
gemini_service.py
=================
Wraps Google Gemini Vision API for weld defect analysis.

Each call receives a **stitched image** (two consecutive frames
joined side-by-side).  The prompt tells Gemini about this layout
so it can reference left vs right sections in its output.

Includes:
  - Weld image validation (reject non-weld images)
  - Rich, precise bounding box instructions
  - Production-grade error handling and JSON salvage
"""
import google.generativeai as genai
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

genai.configure(api_key=settings.GEMINI_API_KEY)

# ---------------------------------------------------------------------------
# Weld validation prompt — fast pre-check before full analysis
# ---------------------------------------------------------------------------
WELD_VALIDATION_PROMPT = """\
You are a weld inspection system. Your ONLY job right now is to determine whether this image shows a welded joint or weld bead on metal.

A valid weld image shows: a visible weld bead/seam on metal base plate, possibly with defects like porosity, undercut, spatter, etc.
An INVALID image is: a person, landscape, document, QR code, screenshot, chart, face, vehicle, food, random object, or anything that is NOT a welded metal joint.

Respond ONLY with this exact JSON — no preamble, no markdown:
{"is_weld_image": true or false, "reason": "<one sentence max>"}
"""

# ---------------------------------------------------------------------------
# Main analysis prompt
# ---------------------------------------------------------------------------
WELD_ANALYSIS_PROMPT = """\
Role: You are a Certified Welding Inspector (CWI) and Quality Assurance Expert performing a visual surface inspection.

Image layout: The image is a STITCHED pair of adjacent weld video frames joined side-by-side. Treat the entire image as ONE continuous weld segment. The left half is Frame A, the right half is Frame B.

════════════════════════════════════════════════════════════
DEFECT IDENTIFICATION CRITERIA (strict colour-coded legend)
════════════════════════════════════════════════════════════

🔴 RED — Blowholes / Surface Porosity
  Visible as: round pinholes, open craters, or clusters of surface pores on the weld face.
  Label format: "Surface Porosity, est. X-Y visible" or "Blowholes (X visible)"
  Severity rule: ≤3 pores = medium; 4-9 = high; ≥10 = critical

🟠 ORANGE — Undercut
  Visible as: a groove or channel melted into the base metal along the weld toe, running parallel to the weld bead.
  Label format: "Extensive Undercut (Top Toe)", "Undercut Segments (Top & Bottom Toes)"
  Note: Specify EXACTLY which toe(s) are affected. Top toe = top edge of weld; Bottom toe = bottom edge.
  Severity rule: <10mm = low; 10-30mm = medium; >30mm = high or critical

🔵 BLUE — Underfill
  Visible as: a depression or valley in the weld face where the weld surface is BELOW the parent metal level.
  Label format: "Notable Underfill Valleys" or "Underfill Depression"
  Severity rule: shallow = low; moderate = medium; deep groove = high

🟣 PURPLE — Excess Reinforcement / High Humps
  Visible as: the weld bead rises excessively above the parent metal level, forming a tall convex hump.
  Label format: "Excess Reinforcement (High Humps)" or "High Hump"
  Severity rule: slight = low; moderate = medium; severe = high

🟢 GREEN — Spatter
  Visible as: scattered droplets of solidified metal around the weld zone on the base plate.
  Label format: "Widespread Spatter (>20 droplets)" or "Spatter (est. >30 droplets)"
  Severity rule: <10 droplets = low; 10-20 = medium; >20 = high; >50 = critical

⚪ OTHER — Any other defect (cracks, burn-through, overlap, arc strike, slag inclusion, incomplete fusion):
  Identify, describe, and map them clearly.

════════════════════════════════════════════════════════════
BOUNDING BOX RULES (CRITICAL — read carefully)
════════════════════════════════════════════════════════════
• All coordinates are RELATIVE (0.0 to 1.0) over the FULL stitched image width and height.
• x=0, y=0 is the TOP-LEFT corner of the full stitched image.
• x=0.5 is the CENTER (the seam between Frame A and Frame B).
• Make bounding boxes TIGHT — they should closely wrap the actual defect region, not the whole image.
• For UNDERCUT: the bounding box should span the full length of the undercut groove along the toe.
  - Top Toe undercut: y should be small (near top of weld, e.g. 0.1 to 0.35)
  - Bottom Toe undercut: y should be larger (near bottom of weld, e.g. 0.55 to 0.85)
• For SPATTER: the bounding box should cover the entire region where spatter droplets are scattered.
• For BLOWHOLES: draw one tight box per cluster of pinholes.
• For EXCESS REINFORCEMENT: box around the high hump regions.
• For UNDERFILL: box along the valley depression.

════════════════════════════════════════════════════════════
OUTPUT FORMAT — Return ONLY valid raw JSON, NO markdown fences
════════════════════════════════════════════════════════════
{
  "overall_result": "pass" | "fail" | "review",
  "weld_quality_score": <0-100, where 100=perfect, 0=completely defective>,
  "total_weld_length_mm": <estimated length in mm or null>,
  "defects": [
    {
      "defect_id": "D001",
      "type": "<exact type: Blowholes, Undercut, Underfill, Excess Reinforcement, Spatter, Overlap, Crack, etc.>",
      "label": "<display label per legend format above, max 50 chars>",
      "severity": "low" | "medium" | "high" | "critical",
      "description": "<precise visual description of what you see, max 120 chars>",
      "confidence": <0.0-1.0>,
      "estimated_count": "<e.g. 'est. 15-20 droplets' or null>",
      "bounding_box": {"x": 0.0, "y": 0.0, "width": 0.0, "height": 0.0},
      "length_mm": <number or null>,
      "depth_mm": <number or null>,
      "width_mm": <number or null>,
      "count": <integer or null>,
      "position": "<left section / right section / spanning both — plus weld zone location>",
      "standards_reference": "<e.g. AWS D1.1 Clause 6.9 or ISO 5817 Level C>",
      "recommendation": "<specific corrective action, max 100 chars>"
    }
  ],
  "defect_summary": {"Blowholes": 1, "Undercut": 2},
  "standards_compliance": [
    {"standard": "AWS D1.1", "grade": null, "compliant": true|false, "notes": "<max 80 chars>"},
    {"standard": "ISO 5817", "grade": "B|C|D", "compliant": true|false, "notes": "<max 80 chars>"}
  ],
  "recommendations": ["<top priority action, max 80 chars>", "<second priority, max 80 chars>"],
  "model_notes": "<left-to-right visual walkthrough of defect locations, max 150 chars>"
}

If NO defects found: empty "defects" array, empty "defect_summary", set "overall_result" to "pass", score ≥ 85.
Scoring guide: deduct points per defect — critical=25pts, high=15pts, medium=8pts, low=3pts. Start at 100.
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

        defect_pattern = re.compile(r'\{[^{}]*"defect_id"[^{}]*\}', re.DOTALL)
        raw_defects    = defect_pattern.findall(raw)
        parsed_defects = []
        for d in raw_defects:
            try:
                parsed_defects.append(json.loads(d))
            except Exception:
                pass

        logger.warning(
            f"Gemini response truncated — salvaged {len(parsed_defects)} defects"
        )
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


class GeminiService:
    def __init__(self):
        self.model = genai.GenerativeModel("gemini-2.5-flash")

    def _call_gemini(self, prompt: str, image_bytes: bytes, mime_type: str) -> str:
        """Raw Gemini call — returns stripped text."""
        import time
        start = time.time()
        response = self.model.generate_content(
            [prompt, {"mime_type": mime_type, "data": image_bytes}],
            generation_config=genai.GenerationConfig(
                temperature=0.1,
                max_output_tokens=8192,
                response_mime_type="application/json",
            ),
        )
        elapsed = round(time.time() - start, 2)
        raw = response.text.strip()
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw.rstrip())
        logger.info(f"Gemini responded in {elapsed}s | {len(raw)} chars")
        return raw

    def validate_weld_image(self, image_bytes: bytes, mime_type: str = "image/jpeg") -> tuple[bool, str]:
        """
        Returns (is_valid_weld, reason).
        Raises HTTPException with 422 if the image is not a weld image.
        """
        try:
            raw = self._call_gemini(WELD_VALIDATION_PROMPT, image_bytes, mime_type)
            data = json.loads(raw)
            is_weld = bool(data.get("is_weld_image", False))
            reason = str(data.get("reason", "Unknown"))
            return is_weld, reason
        except Exception as e:
            # If validation itself fails, be permissive and log
            logger.warning(f"Weld validation check failed (permissive pass): {e}")
            return True, "Validation check skipped"

    def _call_gemini_analysis(self, image_bytes: bytes, mime_type: str) -> dict:
        try:
            raw = self._call_gemini(WELD_ANALYSIS_PROMPT, image_bytes, mime_type)
            return _salvage_truncated_json(raw)
        except Exception as e:
            logger.error(f"Gemini API call failed: {e}")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"AI processing failed: {str(e)}",
            )

    def analyze_pair(
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
    ) -> FramePairResult:
        """Analyze a stitched frame pair and return a FramePairResult."""
        raw = self._call_gemini_analysis(stitched_bytes, mime_type)

        defects = []
        for d in raw.get("defects", []):
            bb = d.get("bounding_box")
            try:
                # Clamp bounding box values to valid range
                if bb:
                    bb = {
                        "x": max(0.0, min(1.0, float(bb.get("x", 0)))),
                        "y": max(0.0, min(1.0, float(bb.get("y", 0)))),
                        "width": max(0.01, min(1.0, float(bb.get("width", 0.1)))),
                        "height": max(0.01, min(1.0, float(bb.get("height", 0.1)))),
                    }
                defects.append(Defect(
                    defect_id=str(d.get("defect_id", str(uuid.uuid4())[:8])),
                    type=str(d.get("type", "Unknown")),
                    label=d.get("label"),
                    severity=DefectSeverity(d.get("severity", "medium")),
                    description=str(d.get("description", "")),
                    confidence=float(d.get("confidence", 0.8)),
                    bounding_box=BoundingBox(**bb) if bb else None,
                    length_mm=d.get("length_mm"),
                    depth_mm=d.get("depth_mm"),
                    width_mm=d.get("width_mm"),
                    count=d.get("count"),
                    estimated_count=d.get("estimated_count"),
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
                continue

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
            overall_result=OverallResult(raw.get("overall_result", "review")),
            weld_quality_score=float(raw.get("weld_quality_score", 0.0)),
            defects=defects,
            defect_summary=raw.get("defect_summary", {}),
            standards_compliance=standards,
            recommendations=raw.get("recommendations", []),
            model_notes=raw.get("model_notes"),
        )


gemini_service = GeminiService()
