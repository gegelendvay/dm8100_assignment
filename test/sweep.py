import subprocess
import re
import statistics
import sys

time_re = re.compile(r"Time\s*\(s\):\s*([\d.]+)")
checksum_re = re.compile(r"Checksum\(C\):\s*([0-9.eE+-]+)")

def parse_task_arg():
    for arg in sys.argv[1:]:
        if arg.startswith("task="):
            return arg.split("=", 1)[1]
    raise SystemExit("Usage: python sweep.py task=X")

def run_and_parse(cmd, runs=100):
    times = []
    checksums = []
    for i in range(runs):
        result = subprocess.run(cmd, capture_output=True, text=True)
        output = result.stdout + result.stderr

        t = time_re.search(output)
        c = checksum_re.search(output)

        if t:
            times.append(float(t.group(1)))
        else:
            print(f"[WARNING] Run {i+1} produced no parseable time. Output: {output.strip()}")

        if c:
            checksums.append(c.group(1))

    return times, checksums

task = parse_task_arg()
cmd = f"./build/task{task}"
runs = 100

print(f"Running {cmd} ({runs} times)...")
times, checksums = run_and_parse(cmd, runs=runs)

if not times:
    raise RuntimeError("No parseable timing output was found.")

avg_time = statistics.mean(times)

print("\n" + "=" * 50)
print(f"Task: {task}")
print(f"Command: {cmd}")
print(f"Samples: {len(times)}/{runs}")
print(f"Average time: {avg_time:.6f}s")
print(f"Min time:     {min(times):.6f}s")
print(f"Max time:     {max(times):.6f}s")
if checksums:
    print(f"Last checksum: {checksums[-1]}")
print("=" * 50)