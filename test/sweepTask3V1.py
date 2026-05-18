import re
import sys
import shlex
import subprocess
import statistics

time_re = re.compile(r"Time:\s*([0-9.eE+-]+)\s*seconds")
checksum_re = re.compile(r"Checksum:\s*([0-9.eE+-]+)")
check_checksum_re = re.compile(r"Check checksum:\s*([0-9.eE+-]+)")


def run_and_parse(cmd, runs=100):
    times = []
    checksums = []
    reference_checksums = []

    for i in range(runs):
        result = subprocess.run(cmd, capture_output=True, text=True)
        output = result.stdout + result.stderr

        t_matches = time_re.findall(output)
        c_matches = checksum_re.findall(output)
        cc_matches = check_checksum_re.findall(output)

        if t_matches:
            times.append(float(t_matches[-1]))
        else:
            print(f"[WARNING] Run {i+1} produced no parseable time.")
            print(output.strip())

        if c_matches:
            checksums.append(float(c_matches[-1]))

        if cc_matches:
            reference_checksums.append(float(cc_matches[-1]))

    return times, checksums, reference_checksums


if len(sys.argv) < 2:
    raise SystemExit("Usage: python3 sweepAna.py [command...]")

if len(sys.argv) == 2:
    cmd = shlex.split(sys.argv[1])
else:
    cmd = sys.argv[1:]

runs = 100

print(f"Running {' '.join(cmd)} ({runs} times)...")

times, checksums, reference_checksums = run_and_parse(cmd, runs)

if not times:
    raise RuntimeError("No parseable timing output was found.")

avg_time = statistics.mean(times)

print("\n" + "=" * 50)
print(f"Command: {' '.join(cmd)}")
print(f"Samples: {len(times)}/{runs}")
print(f"Average time: {avg_time:.6f} s")
print(f"Min time: {min(times):.6f} s")
print(f"Max time: {max(times):.6f} s")

if len(times) > 1:
    print(f"Std dev: {statistics.stdev(times):.6f} s")

if checksums:
    print(f"Last checksum: {checksums[-1]:.10f}")

if reference_checksums:
    print(f"Last reference checksum: {reference_checksums[-1]:.10f}")
    diff = abs(checksums[-1] - reference_checksums[-1])
    print(f"Checksum difference: {diff:.10e}")

print("=" * 50)
