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

// Broadcast the centroids array (length K*D) from `root` to all ranks.
void broadcast_centroids(const double* centroids, int K, int D, int root);

// Server: aggregate local sums & counts from all ranks into global centroids.
// Assumes local_cent (K*D) holds sums, local_cnts (K) holds counts.
void server_aggregate(const double* local_cent,
                      const int*    local_cnts,
                      double*       global_cent,
                      int           K,
                      int           D);

// Worker: send local sums & counts to server, receive new global centroids back.
// After call, `centroids` holds the updated global centroids.
void worker_exchange(double*       centroids,
                     const int*    local_cnts,
                     int           K,
                     int           D,
                     int           server_rank);

// Fully-decentralized: everyone ends up with true global average.
void allreduce_average(double*       centroids,  // IN: local sums; OUT: global avg
                       const int*    counts,     // IN: local counts
                       int           K,
                       int           D);

#endif // MPI_HELPERS_HPP
