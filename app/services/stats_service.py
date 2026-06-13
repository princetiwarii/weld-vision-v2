from collections import defaultdict, Counter
from typing import List
from app.schemas.inspection import (
    FramePairResult, StatisticalSummary, DefectStatEntry, OverallResult,
)


def compute_statistics(results: List[FramePairResult]) -> StatisticalSummary:
    if not results:
        return StatisticalSummary(
            total_frames_analyzed=0, total_defects_found=0,
            defect_type_stats=[], avg_quality_score=0.0,
            min_quality_score=0.0, max_quality_score=0.0,
            pass_count=0, fail_count=0, review_count=0,
            overall_compliance_aws=False, overall_compliance_iso=False,
            top_recommendations=[],
        )

    dtype_data: dict = defaultdict(lambda: {
        "count": 0, "frames": set(), "confidences": [],
        "severities": Counter(), "lengths": [], "depths": [],
    })

    for res in results:
        for d in res.defects:
            t = d.type
            dtype_data[t]["count"]       += 1
            dtype_data[t]["frames"].add(res.frame_index)
            dtype_data[t]["confidences"].append(d.confidence)
            dtype_data[t]["severities"][d.severity.value] += 1
            if d.length_mm is not None:
                dtype_data[t]["lengths"].append(d.length_mm)
            if d.depth_mm is not None:
                dtype_data[t]["depths"].append(d.depth_mm)

    defect_type_stats = []
    for dtype, data in sorted(dtype_data.items(), key=lambda x: -x[1]["count"]):
        confs = data["confidences"]
        lens  = data["lengths"]
        deps  = data["depths"]
        defect_type_stats.append(DefectStatEntry(
            defect_type=dtype,
            total_count=data["count"],
            frames_affected=len(data["frames"]),
            avg_confidence=round(sum(confs) / len(confs), 3) if confs else 0.0,
            severity_breakdown=dict(data["severities"]),
            avg_length_mm=round(sum(lens) / len(lens), 2) if lens else None,
            avg_depth_mm=round(sum(deps) / len(deps), 2) if deps else None,
        ))

    scores       = [r.weld_quality_score for r in results]
    pass_count   = sum(1 for r in results if r.overall_result == OverallResult.PASS)
    fail_count   = sum(1 for r in results if r.overall_result == OverallResult.FAIL)
    review_count = sum(1 for r in results if r.overall_result == OverallResult.REVIEW)

    def _all_comply(standard: str) -> bool:
        for r in results:
            for s in r.standards_compliance:
                if s.standard == standard and not s.compliant:
                    return False
        return True

    severity_rank = {"critical": 4, "high": 3, "medium": 2, "low": 1}
    worst, worst_rank = None, 0
    for r in results:
        for d in r.defects:
            rank = severity_rank.get(d.severity.value, 0)
            if rank > worst_rank:
                worst_rank = rank
                worst = f"{d.type} ({d.severity.value})"

    rec_counter: Counter = Counter()
    for r in results:
        for rec in r.recommendations:
            rec_counter[rec.strip()] += 1
    top_recs = [r for r, _ in rec_counter.most_common(8)]

    return StatisticalSummary(
        total_frames_analyzed=len(results),
        total_defects_found=sum(len(r.defects) for r in results),
        defect_type_stats=defect_type_stats,
        avg_quality_score=round(sum(scores) / len(scores), 2),
        min_quality_score=min(scores),
        max_quality_score=max(scores),
        pass_count=pass_count,
        fail_count=fail_count,
        review_count=review_count,
        most_severe_defect=worst,
        overall_compliance_aws=_all_comply("AWS D1.1"),
        overall_compliance_iso=_all_comply("ISO 5817"),
        top_recommendations=top_recs,
    )
