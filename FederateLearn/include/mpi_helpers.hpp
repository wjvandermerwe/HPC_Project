//
// Created by johan on 17/05/2025.
//

#ifndef MPI_HELPERS_HPP
#define MPI_HELPERS_HPP

#include <mpi.h>
#include <vector>

// Initialize MPI; returns rank and world size by reference.
inline void mpi_initialize(int &rank, int &world_size) {
    MPI_Init(nullptr, nullptr);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
}

// Finalize MPI.
inline void mpi_finalize() {
    MPI_Finalize();
}

void broadcast_centroids(double* centroids, int K, int D, int root = 0);

/*=====  server-worker pattern  =====================================*/
void server_exchange(double*       centroids,   // IN/OUT (only valid on rank 0)
                      const int*    counts,      // IN  (valid on every rank)
                      int           K,
                      int           D);

void worker_exchange(double*       centroids,    // IN  local sums, OUT global avg
                     const int*    local_counts, // IN  local counts
                     int           K,
                     int           D,
                     int           server_rank = 0);

/*=====  peer-to-peer pattern  ======================================*/
void allreduce_average(double* centroids, const int* counts,
                       int K, int D);

#endif // MPI_HELPERS_HPP
