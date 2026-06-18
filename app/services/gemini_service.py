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
# Unified prompt — validation + analysis in one call (token-optimised)
# ---------------------------------------------------------------------------
WELD_INSPECTION_PROMPT = """\
Act as a Certified Welding Inspector (CWI).

Analyze the entire weld image for:
Undercut, Underfill, Excess Reinforcement, Blowholes/Porosity, Spatter.

Identify all visible defects, count occurrences, and provide tight bounding boxes for all defects so our backend can draw them.

Provide highly descriptive, professional `label` values exactly as a CWI would write on a report (e.g. "Extensive Undercut (Top Toe)", "Widespread Spatter (>20 droplets)").

IMPORTANT ANNOTATION INSTRUCTIONS:
You MUST exhaustively identify and provide bounding boxes for EVERY single visible defect. 
Do NOT write the defect name (like "undercut") directly next to the bounding box on the image. Our backend will draw the shapes. Just provide the tight bounding boxes in the JSON output, and include highly descriptive, professional `label` values inside the JSON so they can be displayed in the UI instead of on the image.

Return:
- Defect statistics and summary
- Defect locations (Tight Bounding boxes)
- Severity (low/medium/high/critical)
- Compliance with AWS D1.1 and ISO 5817
- Actionable recommendations

SYSTEM OVERRIDE FOR API INTEGRATION:
To fulfill the requirements for our rendering engine, you MUST output ONLY valid JSON.
Do NOT output markdown. Do NOT output a literal text table. Our Python backend will automatically generate the table and draw the labels directly onto the image using the `bounding_box` coordinates you provide.
CRITICAL: Bounding box coordinates must be precise decimals tightly clamped to the exact visible defect boundary (0.0 to 1.0 relative, where x=0 is left, y=0 is top, x=1 is right, y=1 is bottom).

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
      "label": "<Descriptive Label e.g. 'Extensive Undercut (Top Toe)' or 'Notable Underfill Valleys'>",
      "description": "<Remarks>",
      "severity": "low"|"medium"|"high"|"critical",
      "estimated_count": "<Quantity if applicable>",
      "length_mm": 10.5,
      "bounding_box": {"x": 0.0, "y": 0.0, "width": 0.0, "height": 0.0}
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

        # Allow one level of nested braces for bounding_box
        defect_pattern = re.compile(r'\{\s*"defect_id"[\s\S]*?"bounding_box"\s*:\s*\{[^{}]*\}[\s\S]*?\}', re.DOTALL)
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
        self.model = genai.GenerativeModel("gemini-3.5-flash")

    def _call_gemini(self, prompt: str, image_bytes: bytes, mime_type: str) -> str:
        """Raw Gemini call — returns stripped text with rate limit retries."""
        import time
        from google.api_core.exceptions import ResourceExhausted
        
        for attempt in range(4):
            try:
                start = time.time()
                response = self.model.generate_content(
                    [prompt, {"mime_type": mime_type, "data": image_bytes}],
                    generation_config=genai.GenerationConfig(
                        temperature=0.3,
                        max_output_tokens=8192,
                        response_mime_type="application/json",
                    ),
                )
                elapsed = round(time.time() - start, 2)
                raw = response.text.strip()
                logger.warning(f"RAW GEMINI OUTPUT ({len(raw)} chars):\n{raw}")
                
                # Aggressively extract only the JSON payload in case Gemini generates 
                # conversational text or markdown tables outside the JSON block.
                if "{" in raw and "}" in raw:
                    start_idx = raw.find("{")
                    end_idx = raw.rfind("}") + 1
                    raw = raw[start_idx:end_idx]
                    
                logger.info(f"Gemini responded in {elapsed}s | {len(raw)} chars")
                return raw
            except ResourceExhausted as e:
                if attempt == 3:
                    logger.error(f"Gemini API rate limit exceeded after 4 attempts: {e}")
                    raise
                wait_time = (2 ** attempt) * 5
                logger.warning(f"Gemini Rate Limit hit. Waiting {wait_time}s before retry...")
                time.sleep(wait_time)

    def _call_gemini_unified(self, image_bytes: bytes, mime_type: str) -> dict:
        """
        Single Gemini call that handles both weld validation and full analysis.
        Returns the parsed dict. Raises HTTPException(422) for non-weld images
        and HTTPException(503) for API/parse failures.
        """
        try:
            raw = self._call_gemini(WELD_INSPECTION_PROMPT, image_bytes, mime_type)
            data = _salvage_truncated_json(raw)

            # Gate check: model flagged image as non-weld
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
            logger.error(f"Gemini API call failed: {e}")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"AI processing failed: {str(e)}",
            )

    # Keep as thin shim so callers that still reference validate_weld_image don't break
    def validate_weld_image(self, image_bytes: bytes, mime_type: str = "image/jpeg") -> tuple[bool, str]:
        """Deprecated shim — validation now happens inside _call_gemini_unified."""
        return True, "Validation merged into unified prompt"

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
        raw = self._call_gemini_unified(stitched_bytes, mime_type)

        defects = []
        for d in raw.get("defects", []):
            bb = d.get("bounding_box")
            try:
                if isinstance(bb, list) and len(bb) >= 4:
                    bb = {"x": bb[0], "y": bb[1], "width": bb[2], "height": bb[3]}
                
                # Clamp bounding box values to valid range
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
                if "crit" in sev_raw: sev = "critical"
                elif "high" in sev_raw: sev = "high"
                elif "low" in sev_raw: sev = "low"
                else: sev = "medium"

                # Calculate exact physical length using the bounding box width and known scale
                if length_cm > 0.0 and bb:
                    length_mm = round(bb["width"] * length_cm * 10.0, 1)
                else:
                    raw_len = d.get("length_mm")
                    try:
                        length_mm = float(raw_len) if raw_len is not None and str(raw_len).lower() not in ["n/a", "null", "none"] else None
                    except (ValueError, TypeError):
                        length_mm = None
                    
                est_count = d.get("estimated_count")
                if est_count is not None:
                    est_count = str(est_count)
                
                raw_depth = d.get("depth_mm")
                try:
                    depth_mm = float(raw_depth) if raw_depth is not None and str(raw_depth).lower() not in ["n/a", "null", "none"] else None
                except (ValueError, TypeError):
                    depth_mm = None
                    
                raw_width = d.get("width_mm")
                try:
                    width_mm = float(raw_width) if raw_width is not None and str(raw_width).lower() not in ["n/a", "null", "none"] else None
                except (ValueError, TypeError):
                    width_mm = None

                defects.append(Defect(
                    defect_id=str(d.get("defect_id", str(uuid.uuid4())[:8])),
                    type=str(d.get("type", "Unknown")),
                    label=d.get("label"),
                    severity=DefectSeverity(sev),
                    description=str(d.get("description", "")),
                    confidence=float(d.get("confidence", 0.8)),
                    bounding_box=BoundingBox(**bb) if bb else None,
                    length_mm=length_mm,
                    depth_mm=depth_mm,
                    width_mm=width_mm,
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
                continue

        raw_overall = str(raw.get("overall_result", "review")).lower()
        if "pass" in raw_overall: overall = "pass"
        elif "fail" in raw_overall: overall = "fail"
        else: overall = "review"

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


gemini_service = GeminiService()
