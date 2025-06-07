#!/usr/bin/env python3
"""
run_tests.py – batch-runner for a Nuruomino solver, with
  • per-test timeout  (default 5 s)
  • *.out files used as expected output
  • PASS / FAIL / TIMEOUT report
"""

from pathlib import Path
import subprocess, argparse, sys, time, textwrap, difflib

TIMEOUT_RC = "TIMEOUT"

def run_one(solver: Path, case: Path, timeout: float):
    """Execute solver with `case` piped on stdin; return (elapsed, rc, stdout, stderr)."""
    data = case.read_text(encoding="utf8")
    start = time.perf_counter()

    try:
        proc = subprocess.run(
            [sys.executable, str(solver)],
            input=data,
            text=True,
            capture_output=True,
            timeout=timeout,
        )
        elapsed = time.perf_counter() - start
        return elapsed, str(proc.returncode), proc.stdout, proc.stderr
    except subprocess.TimeoutExpired as exc:
        elapsed = time.perf_counter() - start
        return elapsed, TIMEOUT_RC, exc.stdout or "", exc.stderr or ""


def compare(actual: str, expected: str) -> bool:
    """Whitespace-insensitive comparison (trim lines & drop trailing blank lines)."""
    a_lines = [ln.rstrip() for ln in actual.rstrip().splitlines()]
    e_lines = [ln.rstrip() for ln in expected.rstrip().splitlines()]
    return a_lines == e_lines


def main() -> None:
    ap = argparse.ArgumentParser(description="Run solver on every *.txt test and check against *.out.")
    ap.add_argument("cases_dir", type=Path, help="Folder containing tests.", default=Path("./sample-nuruominoboards"))
    ap.add_argument("--solver", type=Path, default=Path("./proj2425base-nuruomino/nuruomino.py"),)
    ap.add_argument("--timeout", type=float, default=5.0)
    ap.add_argument("-o", "--show-output", action="store_true", help="Print solver stdout / stderr.")
    args = ap.parse_args()

    if not args.solver.exists():
        ap.error(f"Solver '{args.solver}' not found.")

    txt_cases = sorted(f for f in args.cases_dir.rglob("*.txt")
                       if f.is_file() and not f.name.endswith(".out"))

    if not txt_cases:
        ap.error("No *.txt cases found (excluding *.out).")

    totals = dict(run=len(txt_cases), pass_=0, fail=0, timeout=0)
    total_time = 0.0

    for case in txt_cases:
        elapsed, rc, out, err = run_one(args.solver, case, args.timeout)

        expected_file = case.with_suffix(".out")
        if rc == TIMEOUT_RC:
            verdict = "TIMEOUT"
            totals["timeout"] += 1
        elif expected_file.exists():
            exp_text = expected_file.read_text(encoding="utf8")
            if compare(out, exp_text):
                verdict = "PASS"
                totals["pass_"] += 1
                total_time += elapsed
            else:
                verdict = "FAIL"
                totals["fail"] += 1
        else:
            verdict = f"rc={rc}"  # no .out file, just show return code

        print(f"{case.name:<30} {elapsed:>7.3f}s  {verdict}")

        if args.show_output and verdict != "PASS":
            diff = "\n".join(difflib.unified_diff(
                (exp_text if expected_file.exists() else "").splitlines(),
                out.splitlines(),
                fromfile="expected",
                tofile="actual",
                lineterm=""))
            banner = textwrap.dedent(f"""
                ── STDOUT / DIFF ({case.name}) ─────────────────────────────────
                {diff or out.rstrip() or '<no output>'}
                """)
            print(banner)
            if err:
                print(textwrap.indent(err.rstrip(), "stderr: "))

    # summary
    print("\n──── Summary ─────────────────────────")
    print(f"Cases run   : {totals['run']}")
    print(f"Pass        : {totals['pass_']}")
    print(f"Fail        : {totals['fail']}")
    print(f"Timeout     : {totals['timeout']}")
    print(f"Total time  : {total_time:.3f}s")
    print(f"Average     : {total_time/totals['run']:.3f}s")


if __name__ == "__main__":
    main()
