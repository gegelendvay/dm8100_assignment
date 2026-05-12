'''
Run as
- python test/sweep.py './build/task0 <N>' >> test/task0.txt
- python test/sweep.py './build/task1 <N>' >> test/task1.txt
- python test/sweep.py 'mpirun -np <ntasks> --bind-to core --map-by core ./build/task2 <N>' >> test/task2.txt
- python test/sweep.py './build/task3 <N>' >> test/task3.txt

ToDo
- python test/sweep.py 'sbatch run_task2.slurm <nodes> <ntasks-per-node> <N>' >> test/task2.txt
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
threads_re  = re.compile(r"Using\s+(\d+)\s+OpenMP\s+threads")
nodes_re    = re.compile(r"Nodes\s*:\s*(\d+)")
tpn_re      = re.compile(r"Tasks per node\s*:\s*(\d+)")
np_re       = re.compile(r"-np\s+(\d+)")

# ──────────────────────────────────────────────────────────────
# Task detection
# ──────────────────────────────────────────────────────────────

def _detect_task(cmd):
    joined = " ".join(cmd)
    if "task1" in joined: return "task1"
    if "task2" in joined: return "task2"
    if "task3" in joined: return "task3"
    return "task0"

# ──────────────────────────────────────────────────────────────
# Cluster info helpers
# ──────────────────────────────────────────────────────────────

def _get_nodes_and_cpus():
    nodes, cpus = None, None
    try:
        cpus = int(subprocess.run(["nproc"], capture_output=True, text=True).stdout.strip())
    except Exception:
        pass
    try:
        out = subprocess.run(["sinfo", "-o", "%P %c %D"], capture_output=True, text=True).stdout
        for line in out.splitlines():
            parts = line.split()
            if parts[0].startswith("PARTITION") or len(parts) < 3:
                continue
            nodes = int(parts[2])
            if cpus is None:
                cpus = int(parts[1])
            break
    except Exception:
        pass
    return nodes, cpus

def _get_task2_info(cmd, last_output=""):
    """Return (cpus_per_node, nodes) for task2."""
    # sbatch path: job header prints "Nodes: 2" and "Tasks per node: 48"
    m_nodes = nodes_re.search(last_output)
    m_tpn   = tpn_re.search(last_output)
    if m_nodes and m_tpn:
        nodes         = int(m_nodes.group(1))
        cpus_per_node = int(m_tpn.group(1))
        return cpus_per_node, nodes

    # mpirun path: cpus_per_node = -np / nodes (from sinfo)
    joined = " ".join(cmd)
    m_np = np_re.search(joined)
    nodes, _ = _get_nodes_and_cpus()
    if m_np and nodes:
        total_ranks   = int(m_np.group(1))
        cpus_per_node = total_ranks // nodes
        return cpus_per_node, nodes

    return None, nodes

def _get_gpus():
    gpus, name = 0, "Unknown"
    try:
        nsmi = subprocess.run(["nvidia-smi", "-q"], capture_output=True, text=True).stdout
        m = re.search(r"Attached GPUs\s*:\s*(\d+)", nsmi)
        gpus = int(m.group(1)) if m else 0
        m = re.search(r"Product Name\s*:\s*(.+)", nsmi)
        name = m.group(1).strip() if m else "Unknown"
    except Exception:
        pass
    return gpus, name

# ──────────────────────────────────────────────────────────────
# Sweep helpers
# ──────────────────────────────────────────────────────────────

def run_sbatch_and_collect(cmd, poll_interval=1.0):
    result = subprocess.run(cmd, capture_output=True, text=True)
    submit_output = result.stdout + result.stderr

    m = jobid_re.search(submit_output)
    if not m:
        raise RuntimeError(f"Could not parse job id from sbatch output:\n{submit_output}")
    jobid = m.group(1)

    while True:
        status = subprocess.run(
            ["sacct", "-j", jobid, "--format=State", "--noheader"],
            capture_output=True, text=True)
        state_line = status.stdout.strip()
        state = state_line.split()[0] if state_line else ""
        if state in ("COMPLETED", "FAILED", "CANCELLED", "TIMEOUT"):
            break
        time.sleep(poll_interval)

    out_file = f"build/task2_{jobid}.out"
    err_file = f"build/task2_{jobid}.err"
    collected = ""
    if os.path.exists(out_file):
        with open(out_file) as f: collected += f.read()
    if os.path.exists(err_file):
        with open(err_file) as f: collected += ("\n" if collected else "") + f.read()
    for path in (out_file, err_file):
        if os.path.exists(path): os.remove(path)
    return collected

def run_and_parse(cmd, runs=100):
    times, checksums, thread_counts = [], [], []
    last_output = ""

    for i in range(runs):
        if cmd[0] == "sbatch":
            output = run_sbatch_and_collect(cmd)
        else:
            result = subprocess.run(cmd, capture_output=True, text=True)
            output = result.stdout + result.stderr
        last_output = output

        t_matches = time_re.findall(output)
        c_matches = checksum_re.findall(output)
        th_match  = threads_re.search(output)

        if t_matches:
            unit, value = t_matches[-1]
            t = float(value)
            if unit == "ms": t /= 1000.0
            times.append(t)
        else:
            print(f"[WARNING] Run {i+1} produced no parseable time. Output: {output.strip()}")

        if c_matches:  checksums.append(c_matches[-1])
        if th_match:   thread_counts.append(int(th_match.group(1)))

    return times, checksums, thread_counts, last_output

# ──────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────

if len(sys.argv) < 2:
    raise SystemExit("Usage: python sweep.py '<command>' [runs]")

runs = int(sys.argv[-1]) if len(sys.argv) > 2 and sys.argv[-1].isdigit() else 100
cmd  = shlex.split(sys.argv[1])
task = _detect_task(cmd)

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

if task == "task1":
    n_threads = thread_counts[-1] if thread_counts else "N/A"
    print(f"OpenMP threads : {n_threads}")

elif task == "task2":
    cpus_per_node, nodes = _get_task2_info(cmd, last_output)
    print(f"CPU/node       : {cpus_per_node if cpus_per_node is not None else 'N/A'}")
    print(f"Nodes          : {nodes if nodes is not None else 'N/A'}")

elif task == "task3":
    gpus, name = _get_gpus()
    print(f"GPUs/node      : {gpus} ({name})")

print("=" * 50)