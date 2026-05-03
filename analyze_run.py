"""
Analyze the output of a test run against a known expected boat count.

Usage:
    python analyze_run.py test_clips/run_004 --expected 2
    python analyze_run.py test_clips/run_004          # no scoring, just summary
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path


def load_clips(run_dir: Path) -> list[dict]:
    clips = []
    for meta_file in sorted(run_dir.rglob("metadata.json")):
        with open(meta_file) as f:
            data = json.load(f)
        data["_path"] = str(meta_file.parent)
        clips.append(data)
    clips.sort(key=lambda c: c.get("started_at") or "")
    return clips


def parse_dt(s: str | None) -> datetime | None:
    if not s:
        return None
    return datetime.fromisoformat(s)


def fmt_duration(seconds: float | None) -> str:
    if seconds is None:
        return "—"
    m, s = divmod(int(seconds), 60)
    return f"{m}m {s:02d}s" if m else f"{s}s"


def analyze(run_dir: Path, expected: int | None) -> None:
    clips = load_clips(run_dir)

    if not clips:
        print(f"No clips found in {run_dir}")
        return

    print(f"\n{'═' * 62}")
    print(f"  Run: {run_dir}")
    print(f"{'═' * 62}")
    print(f"  {'Track':<8} {'Started':<10} {'Duration':<10} {'Detections':<12} {'Conf mean':<10} {'Error'}")
    print(f"  {'-' * 58}")

    total_detection_frames = 0
    errored = 0

    for c in clips:
        started = parse_dt(c.get("started_at"))
        time_str = started.strftime("%H:%M:%S") if started else "—"
        duration = fmt_duration(c.get("duration_seconds"))
        det = c.get("detection_count", 0)
        conf = f"{c['confidence_mean']:.2f}" if c.get("confidence_mean") else "—"
        error = "⚠" if c.get("error") else ""
        track = f"ID:{c.get('track_id', '?')}"
        print(f"  {track:<8} {time_str:<10} {duration:<10} {det:<12} {conf:<10} {error}")
        total_detection_frames += det
        if c.get("error"):
            errored += 1

    # Gap analysis — group by track_id, flag gaps between consecutive clips of the same track
    print(f"\n  Gap analysis (per track):")
    fragmentation_gaps = []
    by_track: dict[int, list[dict]] = {}
    for c in clips:
        tid = c.get("track_id", -1)
        by_track.setdefault(tid, []).append(c)

    any_gaps = False
    for tid, tclips in sorted(by_track.items()):
        for i in range(1, len(tclips)):
            prev_end = parse_dt(tclips[i - 1].get("ended_at"))
            curr_start = parse_dt(tclips[i].get("started_at"))
            if prev_end and curr_start:
                gap = (curr_start - prev_end).total_seconds()
                if 0 < gap < 120:
                    fragmentation_gaps.append(gap)
                    flag = " ← likely fragmentation" if gap < 30 else ""
                    print(f"    track {tid} gap {i}: {gap:.1f}s{flag}")
                    any_gaps = True
    if not any_gaps:
        print(f"    none")

    print(f"\n  Summary:")
    print(f"    Total clips:          {len(clips)}")
    print(f"    Total detection hits: {total_detection_frames}")
    print(f"    Errored clips:        {errored}")

    if fragmentation_gaps:
        print(f"    Fragmentation gaps:   {len(fragmentation_gaps)} (largest: {max(fragmentation_gaps):.1f}s)")
    else:
        print(f"    Fragmentation gaps:   none")

    if expected is not None:
        print(f"\n  Score:")
        print(f"    Expected boats:  {expected}")
        print(f"    Clips produced:  {len(clips)}")
        diff = len(clips) - expected
        if diff == 0:
            print(f"    Result:          ✓ exact match")
        elif diff > 0:
            print(f"    Result:          ✗ {diff} extra clip(s) — fragmentation or false positives")
        else:
            print(f"    Result:          ✗ {abs(diff)} missing clip(s) — missed detections")

    print(f"{'═' * 62}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze a boat-detector test run")
    parser.add_argument("run_dir", type=Path, help="Path to a test run directory (e.g. test_clips/run_004)")
    parser.add_argument("--expected", "-e", type=int, default=None, metavar="N",
                        help="Expected number of boats for pass/fail scoring")
    args = parser.parse_args()

    if not args.run_dir.exists():
        parser.error(f"Run directory not found: {args.run_dir}")

    analyze(args.run_dir, args.expected)
