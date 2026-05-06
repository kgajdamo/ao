#!/usr/bin/env python3
"""Run all MoE training tests file-by-file with timeouts."""
import subprocess
import re
import json
import os

TEST_DIR = "test/prototype/moe_training"

FILES = [
    "ep/test_a2a_dispatch.py",
    "ep/test_compile.py",
    "ep/test_integration.py",
    "ep/test_kernels.py",
    "ep/test_permute.py",
    "mxfp8/test_mxfp8_a2a.py",
    "test_fp8_grouped_mm.py",
    "test_kernels.py",
    "test_mxfp8_grouped_mm.py",
    "test_nvfp4_grouped_mm.py",
    "test_tensor.py",
    "test_training.py",
]

FILE_TIMEOUTS = {
    "test_kernels.py": 3600,
    "test_mxfp8_grouped_mm.py": 1800,
    "test_tensor.py": 900,
    "test_training.py": 900,
}
DEFAULT_TIMEOUT = 300

def parse_summary(output):
    passed = failed = skipped = xfailed = error = 0
    m = re.search(r'(\d+) passed', output)
    if m: passed = int(m.group(1))
    m = re.search(r'(\d+) failed', output)
    if m: failed = int(m.group(1))
    m = re.search(r'(\d+) skipped', output)
    if m: skipped = int(m.group(1))
    m = re.search(r'(\d+) xfailed', output)
    if m: xfailed = int(m.group(1))
    m = re.search(r'(\d+) error', output)
    if m: error = int(m.group(1))
    return {"passed": passed, "failed": failed + error, "skipped": skipped, "xfailed": xfailed, "total": passed + failed + error + skipped + xfailed}

def run_file(filepath, timeout):
    full_path = f"{TEST_DIR}/{filepath}"
    print(f"\n{'='*60}")
    print(f"FILE: {filepath} (timeout={timeout}s)")
    print(f"{'='*60}", flush=True)
    try:
        r = subprocess.run(
            ["python", "-m", "pytest", full_path, "-v", "--tb=line", "-q"],
            capture_output=True, text=True, timeout=timeout
        )
        output = r.stdout + r.stderr
        lines = output.strip().split("\n")
        for line in lines[-50:]:
            print(line)
        results = parse_summary(output)
        results["status"] = "completed"
        return results, output
    except subprocess.TimeoutExpired:
        print(f"  *** FILE TIMED OUT after {timeout}s ***")
        return {"passed": 0, "failed": 0, "skipped": 0, "xfailed": 0, "total": 0, "status": "timeout"}, ""

def main():
    all_results = {}
    all_outputs = {}
    for filepath in FILES:
        timeout = FILE_TIMEOUTS.get(os.path.basename(filepath), DEFAULT_TIMEOUT)
        results, output = run_file(filepath, timeout)
        all_results[filepath] = results
        all_outputs[filepath] = output
        print(f"  => P:{results['passed']} F:{results['failed']} S:{results['skipped']} X:{results['xfailed']} T:{results['total']} ({results['status']})")

    print(f"\n\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    print(f"{'TEST FILE':<40} {'PASS':>6} {'FAIL':>6} {'SKIP':>6} {'XFAIL':>6} {'TOTAL':>6} {'STATUS':>10}")
    print("-" * 80)
    tp = tf = ts = tx = tt = 0
    for key, r in all_results.items():
        print(f"{key:<40} {r['passed']:>6} {r['failed']:>6} {r['skipped']:>6} {r['xfailed']:>6} {r['total']:>6} {r['status']:>10}")
        tp += r["passed"]; tf += r["failed"]; ts += r["skipped"]; tx += r["xfailed"]; tt += r["total"]
    print("-" * 80)
    print(f"{'TOTAL':<40} {tp:>6} {tf:>6} {ts:>6} {tx:>6} {tt:>6}")

    with open("results/moe_test_results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    # Save full outputs for analysis
    with open("results/moe_test_outputs.json", "w") as f:
        json.dump(all_outputs, f, indent=2)

    print(f"\nResults saved to results/moe_test_results.json")

if __name__ == "__main__":
    main()
