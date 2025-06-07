#!/usr/bin/env python3
"""
run_tests.py – batch-runner for your Nuruomino solver with

  • per-test time-out  (default 5 s)
  • automatic ignore of *.out files
  • optional stdout / stderr dump (-o)

Usage
-----
python run_tests.py  tests/
python run_tests.py  tests/ --timeout 3 --solver nuruomino_fast.py
python run_tests.py  tests/ -o
"""

from pathlib import Path
import subprocess
import argparse
import time
import sys
import textwrap

TIMEOUT_RC = "TIMEOUT"

# ────────────────────────────────────────────────────────────────────────────
# helpers
# ────────────────────────────────────────────────────────────────────────────

def run_one(solver: Path, case: Path, t_limit: float, show_output: bool):
    data = case.read_text(encoding="utf8")

    start = time.perf_counter()
    try:
        proc = subprocess.run(
            [sys.executable, str(solver)],
            input=data,
            text=True,
            capture_output=True,
            timeout=t_limit,
        )
        elapsed = time.perf_counter() - start
        rc = proc.returncode
        stdout, stderr = proc.stdout, proc.stderr
    except subprocess.TimeoutExpired as exc:
        elapsed = time.perf_counter() - start
        rc = TIMEOUT_RC
        stdout, stderr = exc.stdout or "", exc.stderr or ""

    if show_output:
        banner = f"\n===== {case.name} (rc={rc}, {elapsed:.3f}s) ====="
        print(banner)
        if stdout:
            print(textwrap.indent(stdout.rstrip(), "│ "))
        if stderr:
            print(textwrap.indent(stderr.rstrip(), "│ "), file=sys.stderr)

    return elapsed, rc

# ────────────────────────────────────────────────────────────────────────────
# main
# ────────────────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(
        prog="run_tests.py",
        description="Batch-run a Nuruomino solver against .txt cases.",
    )
    p.add_argument("cases_dir", type=Path,
                   help="Directory containing test files (recursively searched).")
    p.add_argument("--solver", type=Path, default=Path("nuruomino.py"),
                   help="Solver script to run (default: ./nuruomino.py).")
    p.add_argument("--timeout", type=float, default=5.0,
                   help="Per-test time limit in seconds (default: 5).")
    p.add_argument("-o", "--show-output", action="store_true",
                   help="Print solver stdout / stderr for each case.")
    args = p.parse_args()

    if not args.solver.exists():
        p.error(f"Solver '{args.solver}' not found.")

    # collect tests:  *.txt  but NOT *.out
    txt_files = sorted(
        f for f in args.cases_dir.rglob("*.txt")
        if f.is_file() and not f.name.endswith(".out")
    )
    if not txt_files:
        p.error("No .txt test files found (*.out are ignored).")

    total_time = 0.0
    failures   = 0

    for case in txt_files:
        elapsed, rc = run_one(args.solver, case, args.timeout, args.show_output)
        total_time += elapsed
        if rc != 0:
            failures += 1
        print(f"{case.name:<30}  {elapsed:>7.3f} s   rc={rc}")

    print("\n──── Summary ─────────────────────────────")
    print(f"Cases run   : {len(txt_files)}")
    print(f"Failures    : {failures}")
    print(f"Total time  : {total_time:.3f} s")
    print(f"Average time: {total_time/len(txt_files):.3f} s")


if __name__ == "__main__":
    main()
