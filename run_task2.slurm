#!/bin/bash

# sinfo -o "%n %C %m"   # per node: name, CPUs (allocated/idle/other/total), memory
# sinfo -o "%P %c %D"   # per partition: name, CPUs-per-node, number of nodes
# sbatch run_task2.slurm

#SBATCH --job-name=dm8100_task2
#SBATCH --nodes=2                        # number of nodes
#SBATCH --ntasks-per-node=64             # MPI ranks per node (set to cores per node)
#SBATCH --time=00:15:00                  # walltime limit
#SBATCH --mem=0                          # 0 = all memory on the node (common default)
#SBATCH --output=build/task2_%j.out
#SBATCH --error=build/task2_%j.err

# Total number of MPI ranks Slurm will launch
NTASKS=${SLURM_NTASKS:-$(( $SLURM_NNODES * $SLURM_NTASKS_PER_NODE ))}

# Choose a matrix size N that is divisible by the number of ranks
N=1024

if (( N % NTASKS != 0 )); then
  echo "Error: N=${N} is not divisible by number of MPI ranks (${NTASKS})."
  exit 1
fi

echo "Running with ${NTASKS} MPI ranks, N=${N}"

# Launch MPI ranks on all cores allocated by Slurm
# Using mpirun inside a Slurm job automatically uses the Slurm allocation.
mpirun ./build/task2 "${N}"
