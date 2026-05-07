'''
Run as
- python test/sweep.py ./build/task1
- python test/sweep.py 'mpirun -np 4 ./build/task2 1024'
- python test/sweep.py 'sbatch run_task2.slurm'
- python test/sweep.py ./build/task3
'''

import os
import re
import sys
import time
import shlex
import subprocess
import statistics

time_s_re = re.compile(r"Time\s*\(s\):\s*([\d.]+)")
time_ms_re = re.compile(r"Time\s*\(ms\):\s*([\d.]+)")
checksum_re = re.compile(r"Checksum\(C\):\s*([0-9.eE+-]+)")
jobid_re = re.compile(r"Submitted batch job (\d+)")

def parse_int(value, default=None):
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return default

def parse_gpu_count_from_string(text):
    if not text:
        return None

    patterns = [
        r"--gpus-per-node(?:=|\s+)(\d+)",
        r"--gpus(?:=|\s+)(\d+)",
        r"--gres(?:=|\s+)gpu(?::[A-Za-z0-9_-]+)?:(\d+)",
        r"--gpus-per-task(?:=|\s+)(\d+)",
    ]
    for pat in patterns:
        m = re.search(pat, text)
        if m:
            return int(m.group(1))
    return None

def infer_resources(cmd):
    env = os.environ

    nodes = parse_int(env.get("SLURM_NNODES"))
    cpus_per_task = parse_int(env.get("SLURM_CPUS_PER_TASK"))
    ntasks = parse_int(env.get("SLURM_NTASKS"))
    gpus = parse_int(env.get("SLURM_GPUS"))

    cmd_str = " ".join(cmd)

    if nodes is None:
        for i, tok in enumerate(cmd):
            if tok.startswith("--nodes="):
                nodes = parse_int(tok.split("=", 1)[1])
                break
            if tok == "--nodes" and i + 1 < len(cmd):
                nodes = parse_int(cmd[i + 1])
                break

    if ntasks is None:
        for i, tok in enumerate(cmd):
            if tok in ("-np", "-n") and i + 1 < len(cmd):
                ntasks = parse_int(cmd[i + 1])
                break
            if tok.startswith("--ntasks="):
                ntasks = parse_int(tok.split("=", 1)[1])
                break
            if tok == "--ntasks" and i + 1 < len(cmd):
                ntasks = parse_int(cmd[i + 1])
                break

    if cpus_per_task is None:
        for i, tok in enumerate(cmd):
            if tok.startswith("--cpus-per-task="):
                cpus_per_task = parse_int(tok.split("=", 1)[1])
                break
            if tok == "--cpus-per-task" and i + 1 < len(cmd):
                cpus_per_task = parse_int(cmd[i + 1])
                break

    if gpus is None:
        gpus = parse_gpu_count_from_string(cmd_str)

    cpu_cores = None
    if cpus_per_task is not None and ntasks is not None:
        cpu_cores = cpus_per_task * ntasks
    elif cpus_per_task is not None:
        cpu_cores = cpus_per_task
    elif ntasks is not None and ("mpirun" in cmd or "srun" in cmd):
        cpu_cores = ntasks

    if nodes is None:
        nodes = 1

    if gpus is None:
        gpus = 0

    return {
        "nodes": nodes,
        "cpus_per_task": cpus_per_task,
        "ntasks": ntasks,
        "cpu_cores": cpu_cores,
        "gpus": gpus,
    }

def read_text_file(path):
    if os.path.exists(path):
        with open(path, "r") as f:
            return f.read()
    return ""

def run_sbatch_and_collect(cmd, poll_interval=1.0):
    """
    Submit a Slurm batch job with sbatch, wait until it finishes,
    read its output/error files, then delete them and return the contents.
    """
    result = subprocess.run(cmd, capture_output=True, text=True)
    submit_output = result.stdout + result.stderr

    m = jobid_re.search(submit_output)
    if not m:
        raise RuntimeError(f"Could not parse job id from sbatch output:\n{submit_output}")

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

    out_candidates = [
        f"build/task2_{jobid}.out",
        f"build/task3_{jobid}.out",
        f"slurm-{jobid}.out",
    ]
    err_candidates = [
        f"build/task2_{jobid}.err",
        f"build/task3_{jobid}.err",
        f"slurm-{jobid}.err",
    ]

    collected = ""
    used_files = []

    for path in out_candidates:
        if os.path.exists(path):
            collected += read_text_file(path)
            used_files.append(path)
            break

    for path in err_candidates:
        if os.path.exists(path):
            err_text = read_text_file(path)
            if err_text:
                if collected:
                    collected += "\n"
                collected += err_text
            used_files.append(path)
            break

    for path in used_files:
        if os.path.exists(path):
            os.remove(path)

    return collected

def run_and_parse(cmd, runs=100):
    times_s = []
    checksums = []

    compile_only = (cmd[0] == "nvcc")

    actual_runs = 1 if compile_only else runs

    for i in range(actual_runs):
        if cmd[0] == "sbatch":
            output = run_sbatch_and_collect(cmd)
        else:
            result = subprocess.run(cmd, capture_output=True, text=True)
            output = result.stdout + result.stderr

        if result.returncode != 0 if cmd[0] != "sbatch" else False:
            print(f"[WARNING] Run {i+1} failed with return code {result.returncode}")
            print(output.strip())

        t_matches_s = time_s_re.findall(output)
        t_matches_ms = time_ms_re.findall(output)
        c_matches = checksum_re.findall(output)

        if t_matches_s:
            times_s.append(float(t_matches_s[-1]))
        elif t_matches_ms:
            times_s.append(float(t_matches_ms[-1]) / 1000.0)
        elif not compile_only:
            print(
                f"[WARNING] Run {i+1} produced no parseable time. "
                f"Output: {output.strip()}"
            )

        if c_matches:
            checksums.append(c_matches[-1])

    return times_s, checksums, compile_only

if len(sys.argv) < 2:
    raise SystemExit("Usage: python sweep.py [args...]")

if len(sys.argv) == 2:
    cmd = shlex.split(sys.argv[1])
else:
    cmd = sys.argv[1:]

runs = 100
resources = infer_resources(cmd)

print(f"Running {' '.join(cmd)} ({1 if cmd[0] == 'nvcc' else runs} times)...")

times, checksums, compile_only = run_and_parse(cmd, runs=runs)

print("\n" + "=" * 50)
print(f"Command: {' '.join(cmd)}")
print(f"Nodes: {resources['nodes']}")
print(f"CPU cores: {resources['cpu_cores'] if resources['cpu_cores'] is not None else 'unknown'}")
print(f"CPUs per task: {resources['cpus_per_task'] if resources['cpus_per_task'] is not None else 'unknown'}")
print(f"MPI tasks: {resources['ntasks'] if resources['ntasks'] is not None else 'unknown'}")
print(f"GPU cores: {resources['gpus']}")

if compile_only:
    print("Mode: compile only")
else:
    if not times:
        raise RuntimeError("No parseable timing output was found.")
    avg_time = statistics.mean(times)
    print(f"Samples: {len(times)}/{runs}")
    print(f"Average time: {avg_time:.6f}s")
    print(f"Min time: {min(times):.6f}s")
    print(f"Max time: {max(times):.6f}s")

if checksums:
    print(f"Last checksum: {checksums[-1]}")
print("=" * 50)
