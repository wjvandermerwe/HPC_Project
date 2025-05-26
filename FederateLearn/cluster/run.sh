#!/bin/bash
#SBATCH --job-name=kmeans_mpi
#SBATCH --partition=compute          # ← replace with your partition/queue
#SBATCH --nodes=8                    # total nodes
#SBATCH --ntasks-per-node=1          # one MPI rank per node
#SBATCH --cpus-per-task=8            # threads per rank (if you use OpenMP)
#SBATCH --time=01:00:00              # HH:MM:SS wall-clock limit
#SBATCH --output=slurm-%j.out        # stdout+stderr log

module load gcc openmpi              # or whichever toolchain your site uses
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

srun ./run_mpi                    # your compiled executable name
