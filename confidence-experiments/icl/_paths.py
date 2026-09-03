"""Paths for the ICL experiments. common.py is imported from the original pilot
rather than forked: it is the measurement layer (token-level confidence,
teacher-forced scoring, grading), and changing it would break comparability."""
import sys
from pathlib import Path

PILOT = Path(__file__).resolve().parents[2] / "basic-tests-updated"
SEQ = PILOT / "outputs" / "qwen17b_sequential_seed42"
TRACES = SEQ / "06_sequential_trace.jsonl"
SUMMARY = SEQ / "06_summary.json"
if str(PILOT) not in sys.path:
    sys.path.insert(0, str(PILOT))
