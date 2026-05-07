'''
Run as
- python test/sweep.py ./build/task1 >> test/task1.txt
- python test/sweep.py 'mpirun -np 4 ./build/task2 1024' >> test/task2.txt
- python test/sweep.py 'sbatch ../run_task2.slurm' >> test/task2.txt
'''

import os
import re
import sys
import time
import shlex
import subprocess
import statistics

time_re = re.compile(r"Time\s*\(s\):\s*([\d.]+)")
checksum_re = re.compile(r"Checksum\(C\):\s*([0-9.eE+-]+)")
jobid_re = re.compile(r"Submitted batch job (\d+)")


def run_sbatch_and_collect(cmd, poll_interval=1.0):
    """
    Submit a Slurm batch job with sbatch, wait until it finishes,
    read its output/error files, then delete them and return the contents.
    """
    # Submit job
    result = subprocess.run(cmd, capture_output=True, text=True)
    submit_output = result.stdout + result.stderr

    m = jobid_re.search(submit_output)
    if not m:
        raise RuntimeError(f"Could not parse job id from sbatch output:\n{submit_output}")

    jobid = m.group(1)

    # Wait for job to finish; adjust command/states to match your cluster
    while True:
        # Example using sacct; if unavailable, switch to squeue-based polling
        status = subprocess.run(
            ["sacct", "-j", jobid, "--format=State", "--noheader"],
            capture_output=True,
            text=True,
        )
        state_line = status.stdout.strip()
        state = state_line.split()[0] if state_line else ""

        if state in ("COMPLETED", "FAILED", "CANCELLED", "TIMEOUT"):
            break

        time.sleep(poll_interval)

    # Collect Slurm output from run_task2.slurm
    out_file = f"build/task2_{jobid}.out"
    err_file = f"build/task2_{jobid}.err"

    collected = ""
    if os.path.exists(out_file):
        with open(out_file, "r") as f:
            collected += f.read()
    if os.path.exists(err_file):
        with open(err_file, "r") as f:
            if collected:
                collected += "\n"
            collected += f.read()

    # Delete the files once their contents have been collected
    for path in (out_file, err_file):
        if os.path.exists(path):
            os.remove(path)

    return collected


def run_and_parse(cmd, runs=100):
    times = []
    checksums = []

    for i in range(runs):
        # Special case: sbatch command, which produces Slurm files instead of direct stdout
        if cmd[0] == "sbatch":
            output = run_sbatch_and_collect(cmd)
        else:
            result = subprocess.run(cmd, capture_output=True, text=True)
            output = result.stdout + result.stderr

        # Get all matches and take the last one
        t_matches = time_re.findall(output)
        c_matches = checksum_re.findall(output)

        if t_matches:
            times.append(float(t_matches[-1]))
        else:
            print(
                f"[WARNING] Run {i+1} produced no parseable time. "
                f"Output: {output.strip()}"
            )

        if c_matches:
            checksums.append(c_matches[-1])

    return times, checksums


if len(sys.argv) < 2:
    raise SystemExit("Usage: python sweep.py [args...]")

# Support two styles:
# 1) python sweep.py mpirun -np 4 ./build/task2 1024
# 2) python sweep.py 'mpirun -np 4 ./build/task2 1024'
if len(sys.argv) == 2:
    # Single string: split it like a shell command line
    cmd = shlex.split(sys.argv[1])
else:
    # Already split by the shell into program + args
    cmd = sys.argv[1:]

runs = 100

print(f"Running {' '.join(cmd)} ({runs} times)...")
times, checksums = run_and_parse(cmd, runs=runs)

if not times:
    raise RuntimeError("No parseable timing output was found.")

avg_time = statistics.mean(times)

print("\n" + "=" * 50)
print(f"Command: {' '.join(cmd)}")
print(f"Samples: {len(times)}/{runs}")
print(f"Average time: {avg_time:.6f}s")
print(f"Min time: {min(times):.6f}s")
print(f"Max time: {max(times):.6f}s")
if checksums:
    print(f"Last checksum: {checksums[-1]}")
print("=" * 50)
