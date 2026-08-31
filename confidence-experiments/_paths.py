"""Shared paths. common.py is imported from the original pilot rather than
forked: it is the measurement layer (token-level confidence extraction,
teacher-forced gold scoring, grading), and changing it would break
comparability with every previously published number."""
import sys
from pathlib import Path

PILOT = Path(__file__).resolve().parent.parent / "basic-tests-updated"
SEQ = PILOT / "outputs" / "qwen17b_sequential_seed42"
TRACES = SEQ / "06_sequential_trace.jsonl"
SUMMARY = SEQ / "06_summary.json"
if str(PILOT) not in sys.path:
    sys.path.insert(0, str(PILOT))
