#!/bin/bash
#SBATCH --job-name=kmeans_mpi
#SBATCH --nodes=5                    # total nodes
#SBATCH --ntasks-per-node=1          # one MPI rank per node
#SBATCH --cpus-per-task=8            # threads per rank (if you use OpenMP)
#SBATCH --time=00:05:00              # HH:MM:SS wall-clock limit
#SBATCH --output=kmeans-%j.out        # combined stdout+stderr

# export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

mpiexec -n $SLURM_NTASKS ./federated_kmeans

mpiexec -n $SLURM_NTASKS ./eval
