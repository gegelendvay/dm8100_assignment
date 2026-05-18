import os
import re
import sys
import time
import shlex
import subprocess
import statistics

kernel_ms_re = re.compile(r"Kernel time:\s*([0-9.eE+-]+)\s*ms")
checksum_re = re.compile(r"Checksum:\s*([0-9.eE+-]+)")
check_checksum_re = re.compile(r"Check checksum:\s*([0-9.eE+-]+)")
jobid_re = re.compile(r"Submitted batch job (\d+)")


def run_sbatch_and_collect(cmd, poll_interval=1.0):
    result = subprocess.run(cmd, capture_output=True, text=True)
    submit_output = result.stdout + result.stderr

    m = jobid_re.search(submit_output)
    if not m:
        raise RuntimeError(f"Could not parse job id:\n{submit_output}")

    jobid = m.group(1)

    while True:
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

    collected = ""

    possible_files = [
        f"cuda_{jobid}.out",
        f"cuda_{jobid}.err",
        f"slurm-{jobid}.out",
        f"build/cuda_{jobid}.out",
        f"build/cuda_{jobid}.err",
    ]

    for path in possible_files:
        if os.path.exists(path):
            with open(path, "r") as f:
                collected += f.read() + "\n"
            os.remove(path)

    return collected


def run_and_parse(cmd, runs=100):
    times_ms = []
    checksums = []
    reference_checksums = []

    for i in range(runs):
        if cmd[0] == "sbatch":
            output = run_sbatch_and_collect(cmd)
        else:
            result = subprocess.run(cmd, capture_output=True, text=True)
            output = result.stdout + result.stderr

        t_matches = kernel_ms_re.findall(output)
        c_matches = checksum_re.findall(output)
        cc_matches = check_checksum_re.findall(output)

        if t_matches:
            times_ms.append(float(t_matches[-1]))
        else:
            print(f"[WARNING] Run {i+1} produced no parseable kernel time.")
            print(output.strip())

        if c_matches:
            checksums.append(float(c_matches[-1]))

        if cc_matches:
            reference_checksums.append(float(cc_matches[-1]))

    return times_ms, checksums, reference_checksums


if len(sys.argv) < 2:
    raise SystemExit("Usage: python test/sweep_cuda.py [command...]")

if len(sys.argv) == 2:
    cmd = shlex.split(sys.argv[1])
else:
    cmd = sys.argv[1:]

runs = 100

print(f"Running {' '.join(cmd)} ({runs} times)...")

times_ms, checksums, reference_checksums = run_and_parse(cmd, runs)

if not times_ms:
    raise RuntimeError("No parseable timing output was found.")

avg_ms = statistics.mean(times_ms)

print("\n" + "=" * 50)
print(f"Command: {' '.join(cmd)}")
print(f"Samples: {len(times_ms)}/{runs}")
print(f"Average kernel time: {avg_ms:.6f} ms")
print(f"Average kernel time: {avg_ms / 1000.0:.6f} s")
print(f"Min kernel time: {min(times_ms):.6f} ms")
print(f"Max kernel time: {max(times_ms):.6f} ms")

if len(times_ms) > 1:
    print(f"Std dev: {statistics.stdev(times_ms):.6f} ms")

if checksums:
    print(f"Last checksum: {checksums[-1]:.10f}")

if reference_checksums:
    print(f"Last reference checksum: {reference_checksums[-1]:.10f}")
    diff = abs(checksums[-1] - reference_checksums[-1])
    print(f"Checksum difference: {diff:.10e}")

print("=" * 50)
