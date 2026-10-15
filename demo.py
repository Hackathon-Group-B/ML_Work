"""Live demo: generate a week of bank data with planted incidents, scan EVERY route blind,
and grade the detector against the hidden answer key, before and after the fix.

Run with:  python demo.py      (Windows: py -3.13 demo.py)

What this is, honestly: a simplified batch scan. It slides a 45-minute window across the
week on each of the 7 routes and applies the detector's statistical test. It is NOT the full
live pipeline (30-second ticks, 5-minute windows, 2 confirmations in a row).
"""
from dataclasses import replace

import pandas as pd

from ml.config import load_config
from ml.detector.scan import PARTS, STEP, WINDOW, fire_windows, merge_into_episodes, window_counts
from ml.generator.build import generate_dataset

SEP = "=" * 72


def verdict(route, ep, labels):
    """Compare a flagged episode with the answer key."""
    for l in labels:
        if l.target.get("route") == route and ep["start"] < l.end and ep["end"] > l.start:
            return "CORRECT - real incident" if l.is_incident else "FALSE ALARM - planted hard negative"
    return "FALSE ALARM - nothing planted here"


def run_rule(title, det, counts_by_route, cfg, labels):
    print(f"\n{title}")
    flagged = []
    for route, (starts, counts) in counts_by_route.items():
        fired = fire_windows(starts, counts, cfg.routes[route].baseline_fail, det, WINDOW)
        flagged += [(route, ep) for ep in merge_into_episodes(fired)]
    caught = false_alarms = 0
    if not flagged:
        print("    Nothing flagged.")
    for route, ep in sorted(flagged, key=lambda x: x[1]["start"]):
        v = verdict(route, ep, labels)
        caught += v.startswith("CORRECT")
        false_alarms += v.startswith("FALSE")
        print(f"    {route}: {ep['start']:%a %d %b %H:%M} to {ep['end']:%H:%M} UTC, "
              f"peak z={ep['peak_z']:.1f}, {ep['peak_ratio']:.1f}x normal -> {v}")
    real = sum(1 for l in labels if l.is_incident)
    print(f"    SCORE: {caught} of {real} real incidents caught, {false_alarms} false alarms")
    return caught, false_alarms


def main():
    print(SEP)
    print("DALATRUST ML DEMO")
    print(SEP)

    cfg = load_config()
    print("\nStep 1: Generating one week of synthetic bank transaction data...")
    df, labels = generate_dataset(cfg)
    print(f"  -> {len(df):,} transactions, 7 days, 12 banks, {df.route.nunique()} routes.")

    print("\nStep 2: ANSWER KEY (hidden from the detector; used only to grade it):")
    for l in labels:
        kind = "REAL INCIDENT" if l.is_incident else "HARD NEGATIVE (should NOT be flagged)"
        print(f"    [{kind}] route {l.target['route']}, {l.start:%a %d %b %H:%M} to {l.end:%H:%M} UTC")

    print(f"\nStep 3: Scanning ALL {len(cfg.routes)} routes blind "
          f"({int(WINDOW.total_seconds() // 60)}-minute windows, sliding every "
          f"{int(STEP.total_seconds() // 60)} minutes)...")
    t0, t1 = df.timestamp.min(), df.timestamp.max()
    counts_by_route = {r: window_counts(df, r, t0, t1, WINDOW, STEP, PARTS) for r in cfg.routes}
    total = sum(len(v[0]) for v in counts_by_route.values())
    print(f"  -> {total:,} windows checked in total.")

    original = replace(cfg.detector, z_min=4.0, confirm_parts=1, confirm_min_parts=1)
    c1, f1 = run_rule("Step 4a: ORIGINAL rule (z >= 4, pooled window only)",
                      original, counts_by_route, cfg, labels)
    c2, f2 = run_rule("Step 4b: IMPROVED rule (z >= 5, and the rate must be elevated in all "
                      "3 separate slices of the window)", cfg.detector, counts_by_route, cfg, labels)

    print("\nSummary:")
    print(f"    Original: {c1} caught, {f1} false alarms")
    print(f"    Improved: {c2} caught, {f2} false alarms")
    print("\nNote: simplified batch scan, one simulated week. Run  python sweep.py  for the")
    print("40-week test of the trade-off (catches vs false alarms).")
    print(SEP)


if __name__ == "__main__":
    main()
