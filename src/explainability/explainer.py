"""
Plain-English Explainability Engine for AIOps Log Anomaly Detection.
Translates neural and volumetric anomaly signals into actionable, human-readable diagnostics.
"""

from typing import Any, Dict, List, Optional
from src.parsing.drain_parser import DrainParser
from src.preprocessing.sessionizer import LogSession


class AnomalyExplainer:
    """Generates plain-English narrative diagnostic explanations for detected alerts."""

    def __init__(self, drain_parser: DrainParser):
        self.drain_parser = drain_parser

    def generate_explanation(
        self,
        session: LogSession,
        score_data: Dict[str, Any],
        is_novel_present: bool = False,
        novel_templates: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Synthesize plain-English diagnostic summary and collect suspicious logs.
        """
        explanations = []
        suspicious_events = []

        seq_score = score_data.get("sequence_score", 0.0)
        ae_score = score_data.get("ae_score", 0.0)
        calibrated_risk = score_data.get("calibrated_risk", 0.0)
        seq_diag = score_data.get("seq_diagnostics", [])
        ae_diag = score_data.get("ae_diagnostics", [])

        # 1. Sequential Transition Explanations
        if seq_diag:
            transition_bullets = []
            for item in seq_diag[:3]:  # Top 3 transitions
                step_idx = item["step_index"]
                actual_eid = item["actual_event"]
                actual_prob = item["actual_prob"] * 100.0
                expected_topk = item.get("expected_topk", [])

                actual_tmpl = self.drain_parser.get_template(actual_eid) or actual_eid
                expected_desc = ", ".join(
                    f"'{self.drain_parser.get_template(t['event_id']) or t['event_id']}' ({t['prob']*100:.1f}%)"
                    for t in expected_topk[:2]
                )

                # Link back to raw log record if available
                raw_line = ""
                log_level = "UNKNOWN"
                ts = ""
                if step_idx < len(session.records):
                    rec = session.records[step_idx]
                    raw_line = rec.raw_text
                    log_level = rec.level
                    ts = rec.timestamp

                transition_bullets.append(
                    f"Step {step_idx + 1}: Unexpected event '{actual_tmpl}' (likelihood: {actual_prob:.2f}%). "
                    f"System expected: {expected_desc}."
                )

                suspicious_events.append({
                    "step": step_idx + 1,
                    "event_id": actual_eid,
                    "template": actual_tmpl,
                    "timestamp": ts,
                    "level": log_level,
                    "raw_log": raw_line,
                    "reason": f"Low transition probability ({actual_prob:.2f}% vs expected {expected_desc})",
                })

            explanations.append(
                f"Abnormal Event Sequence: Detected {len(seq_diag)} unusual execution transitions. "
                + " ".join(transition_bullets)
            )

        # 2. Volumetric Frequency Explanations
        if ae_diag:
            vol_bullets = []
            for dev in ae_diag:
                eid = dev["event_id"]
                actual = dev["actual_count"]
                expected = dev["expected_count"]
                tmpl = self.drain_parser.get_template(eid) or eid

                if actual > expected * 2.0:
                    vol_bullets.append(
                        f"Template '{tmpl}' surged to {actual:.0f} occurrences (typical baseline: ~{expected:.1f})."
                    )
                elif actual < expected * 0.2:
                    vol_bullets.append(
                        f"Expected template '{tmpl}' appeared only {actual:.0f} times (typical baseline: ~{expected:.1f})."
                    )

            if vol_bullets:
                explanations.append("Volumetric Discrepancy: " + " ".join(vol_bullets))

        # 3. Novel / Drifted Template Explanations
        if is_novel_present or novel_templates:
            tmpls = novel_templates or ["<Unseen Event>"]
            explanations.append(
                f"Concept Drift / Novel Event: Observed {len(tmpls)} previously unseen log templates "
                f"({', '.join(tmpls[:2])}) that were not present in normal baseline behavior."
            )

        # Fallback if high score but no granular transition triggered
        if not explanations:
            if calibrated_risk >= 50.0:
                explanations.append(
                    f"Elevated combined anomaly risk ({calibrated_risk:.1f}/100) across neural sequence and "
                    f"reconstruction models exceeding operational tolerance."
                )
            else:
                explanations.append("Session exhibits normal system execution patterns.")

        full_narrative = " \n\n".join(explanations)
        return {
            "summary_text": full_narrative,
            "suspicious_events": suspicious_events,
            "is_sequential_issue": len(seq_diag) > 0,
            "is_volumetric_issue": len(ae_diag) > 0,
            "is_novel_issue": bool(is_novel_present or novel_templates),
        }
