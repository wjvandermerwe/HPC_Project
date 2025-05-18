//
// Created by johan on 17/05/2025.
//
#include "mpi_helpers.hpp"
#include <cstring>  // for memset, memcpy
#include <vector>

void broadcast_centroids(double* centroids, int K, int D, int root) {
    MPI_Bcast((void*)centroids, K*D, MPI_DOUBLE, root, MPI_COMM_WORLD);
}

void server_exchange(double* centroids,
                      const int* counts,
                      int K, int D)
{
    std::vector<double> sum_cent(K * D, 0.0);
    std::vector<int>    sum_cnt(K, 0);

    MPI_Reduce(centroids, sum_cent.data(), K*D, MPI_DOUBLE, MPI_SUM,
               0, MPI_COMM_WORLD);
    MPI_Reduce(counts,    sum_cnt.data(),  K,   MPI_INT,    MPI_SUM,
               0, MPI_COMM_WORLD);

    for (int k = 0; k < K; ++k) {
        double inv = sum_cnt[k] > 0 ? 1.0 / sum_cnt[k] : 0.0;
        for (int d = 0; d < D; ++d) {
            centroids[k*D + d] = sum_cent[k*D + d] * inv;
        }
    }

    // Broadcast the updated centroids back to everyone
    MPI_Bcast(centroids, K*D, MPI_DOUBLE, 0, MPI_COMM_WORLD);
}

void worker_exchange(double* centroids,
                     const int* local_counts,
                     int K, int D,
                     int server_rank)
{
    int rank; MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // Workers send their data; server does the averaging & broadcast.
    MPI_Reduce(centroids, nullptr, K*D, MPI_DOUBLE, MPI_SUM,
               server_rank, MPI_COMM_WORLD);
    MPI_Reduce(local_counts, nullptr, K, MPI_INT, MPI_SUM,
               server_rank, MPI_COMM_WORLD);

    // All ranks (including the server) receive the broadcast.
    MPI_Bcast(centroids, K*D, MPI_DOUBLE, server_rank, MPI_COMM_WORLD);
}

void allreduce_average(double* centroids,
                       const int* counts,
                       int K, int D)
{
    std::vector<double> sum_cent(K*D);
    std::vector<int>    sum_cnt(K);

    // Sum local weighted-sums and counts across all ranks
    MPI_Allreduce(centroids, sum_cent.data(), K*D, MPI_DOUBLE, MPI_SUM,
                  MPI_COMM_WORLD);
    MPI_Allreduce(counts,    sum_cnt.data(),  K,   MPI_INT,    MPI_SUM,
                  MPI_COMM_WORLD);

    // Convert summed weighted-sums to averages
    for (int k = 0; k < K; ++k) {
        double inv = sum_cnt[k] ? 1.0 / sum_cnt[k] : 0.0;
        for (int d = 0; d < D; ++d) {
            centroids[k*D + d] = sum_cent[k*D + d] * inv;
        }
    }
}
