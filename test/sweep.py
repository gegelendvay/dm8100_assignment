'''
Run as
- python test/sweep.py './build/task0 <N>' >> test/task0.txt
- python test/sweep.py './build/task1 <N>' >> test/task1.txt
- python test/sweep.py 'mpirun -np <ntasks> --bind-to core --map-by core ./build/task2 <N>' >> test/task2.txt
- python test/sweep.py './build/task3 <N> <sm_multiplier>' >> test/task3.txt
'''

import os
import re
import sys
import time
import shlex
import subprocess
import statistics

time_re     = re.compile(r"Time\s*\((s|ms)\):\s*([\d.]+)")
checksum_re = re.compile(r"Checksum\(C\):\s*([0-9.eE+-]+)")
jobid_re    = re.compile(r"Submitted batch job (\d+)")

# ──────────────────────────────────────────────────────────────
# Sweep helpers
# ──────────────────────────────────────────────────────────────

def run_and_parse(cmd, runs=100):
    times, checksums, thread_counts = [], [], []
    last_output = ""

    for i in range(runs):
        result = subprocess.run(cmd, capture_output=True, text=True)
        output = result.stdout + result.stderr
        last_output = output

        t_matches = time_re.findall(output)
        c_matches = checksum_re.findall(output)

        if t_matches:
            unit, value = t_matches[-1]
            t = float(value)
            if unit == "ms": t /= 1000.0
            times.append(t)
        else:
            print(f"[WARNING] Run {i+1} produced no parseable time. Output: {output.strip()}")

        if c_matches:  checksums.append(c_matches[-1])

    return times, checksums, thread_counts, last_output

# ──────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────

if len(sys.argv) < 2:
    raise SystemExit("Usage: python sweep.py '<command>' [runs]")

runs = int(sys.argv[-1]) if len(sys.argv) > 2 and sys.argv[-1].isdigit() else 100
cmd  = shlex.split(sys.argv[1])

print(f"Running {' '.join(cmd)} ({runs} times)...")
times, checksums, thread_counts, last_output = run_and_parse(cmd, runs=runs)

if not times:
    raise RuntimeError("No parseable timing output was found.")

avg_time = statistics.mean(times)

print(f"Average time   : {avg_time:.6f}s")
print(f"Min time       : {min(times):.6f}s")
print(f"Max time       : {max(times):.6f}s")
if checksums:
    print(f"Last checksum  : {checksums[-1]}")

print("=" * 50)
