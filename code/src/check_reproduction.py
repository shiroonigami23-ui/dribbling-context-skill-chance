"""Compare a fresh run (code/results/) with the committed results (../results/).

Every JSON number and every numeric CSV column is compared with a relative tolerance (default 1%,
absolute 1e-6), because bootstrap and resampling steps use fixed seeds but library versions can
move the last digits.  Prints a summary and exits non-zero if anything differs beyond tolerance.
Usage (from code/):  python src/check_reproduction.py [--rtol 0.01] [--files summary.json ...]
"""
import argparse
import json
import math
import sys
from pathlib import Path

import pandas as pd

NEW, OLD = Path("results"), Path("../results")


def flat(x, p=""):
    if isinstance(x, dict):
        for k, v in x.items():
            yield from flat(v, f"{p}.{k}" if p else str(k))
    elif isinstance(x, list):
        for i, v in enumerate(x):
            yield from flat(v, f"{p}[{i}]")
    elif isinstance(x, (int, float)) and not isinstance(x, bool):
        yield p, float(x)


def close(a, b, rtol):
    if math.isnan(a) and math.isnan(b):
        return True
    return abs(a - b) <= max(rtol * max(abs(a), abs(b)), 1e-6)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--rtol", type=float, default=0.01)
    ap.add_argument("--files", nargs="*", help="only these result files (default: all)"); a = ap.parse_args()
    bad, checked, missing = [], 0, []
    for old in sorted(OLD.glob("*")):
        new = NEW / old.name
        if old.suffix not in {".json", ".csv"} or (a.files and old.name not in a.files):
            continue
        if not new.exists():
            missing.append(old.name); continue
        if old.suffix == ".json":
            o, n = dict(flat(json.load(open(old)))), dict(flat(json.load(open(new))))
            for k, v in o.items():
                checked += 1
                if k not in n or not close(v, n[k], a.rtol):
                    bad.append((old.name, k, v, n.get(k)))
        else:
            o, n = pd.read_csv(old), pd.read_csv(new)
            if o.shape != n.shape:
                bad.append((old.name, "shape", o.shape, n.shape)); continue
            for c in o.select_dtypes("number").columns:
                for i, (x, y) in enumerate(zip(o[c], n[c])):
                    checked += 1
                    if not close(float(x), float(y), a.rtol):
                        bad.append((old.name, f"{c}[{i}]", x, y))
    print(f"checked {checked} numbers; {len(bad)} differ beyond rtol={a.rtol}; {len(missing)} files missing")
    for m in missing:
        print("  missing:", m)
    by = {}
    for f, k, x, y in bad:
        by.setdefault(f, []).append((k, x, y))
    for f, rows in by.items():
        print(f"  {f}: {len(rows)} differences, e.g. " + "; ".join(f"{k}: {x} -> {y}" for k, x, y in rows[:3]))
    sys.exit(1 if bad or missing else 0)


if __name__ == "__main__":
    main()
