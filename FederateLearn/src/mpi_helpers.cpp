//
// Created by johan on 17/05/2025.
//
#include "mpi_helpers.hpp"
#include <cstring>  // for memset, memcpy
#include <vector>

void broadcast_centroids(const double* centroids, int K, int D, int root) {
    void * buffer = (void*)centroids; // explicit cast stream
    MPI_Bcast(buffer, K*D, MPI_DOUBLE, root, MPI_COMM_WORLD);
}

void server_aggregate(const double* local_cent,
                      const int*    local_cnts,
                      double*       global_cent,
                      int           K,
                      int           D)
{
    // Buffer to accumulate weighted sums on server
    std::vector<double> accum(K * D, 0.0);
    std::vector<int>    cnts(K, 0);

    // Gather all local centroids sums and counts
    MPI_Reduce(local_cent, accum.data(), K*D, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(local_cnts, cnts.data(),    K,   MPI_INT,    MPI_SUM, 0, MPI_COMM_WORLD);

    if (true) { // only rank 0 enters
        for (int k = 0; k < K; ++k) {
            double inv = cnts[k] > 0 ? 1.0 / cnts[k] : 0.0;
            for (int d = 0; d < D; ++d) {
                global_cent[k*D + d] = accum[k*D + d] * inv;
            }
        }
    }
}

void worker_exchange(double* centroids,
                     const int*    local_cnts,
                     int           K,
                     int           D,
                     int           server_rank
                     )
{
    std::vector<double> recv_cent(K * D);
    std::vector<int>    recv_cnts(K);
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // Workers send their sums (in centroids buffer) and counts,
    // and receive the server’s broadcast back as new centroids.
    MPI_Reduce(centroids, recv_cent.data(), K*D, MPI_DOUBLE, MPI_SUM, server_rank, MPI_COMM_WORLD);
    MPI_Reduce(local_cnts, recv_cnts.data(),    K, MPI_INT,    MPI_SUM, server_rank, MPI_COMM_WORLD);

    // On server rank we have to compute the average then broadcast.
    if (rank == server_rank) {
        for (int k = 0; k < K; ++k) {
            double inv = recv_cnts[k] > 0 ? 1.0 / recv_cnts[k] : 0.0;
            for (int d = 0; d < D; ++d) {
                recv_cent[k*D + d] *= inv;
            }
        }
    }
    // Broadcast the updated centroids back to everyone
    MPI_Bcast(recv_cent.data(), K*D, MPI_DOUBLE, server_rank, MPI_COMM_WORLD);
    // Replace local buffer
    std::memcpy(centroids, recv_cent.data(), sizeof(double)*K*D);
}

void allreduce_average(double* centroids, const int* counts, int K, int D) {
    // accumulators
    std::vector<double> sum_cent(K*D);
    std::vector<int>    sum_cnt(K);

    // compute local weighted-sum in centroids itself and share counts
    MPI_Allreduce(centroids, sum_cent.data(), K*D, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(counts,    sum_cnt.data(),  K,   MPI_INT,    MPI_SUM, MPI_COMM_WORLD);

    // on every rank, divide to get the true global avg
    for (int k = 0; k < K; ++k) {
        double inv = sum_cnt[k] > 0 ? 1.0 / sum_cnt[k] : 0.0;
        for (int d = 0; d < D; ++d) {
            centroids[k*D + d] = sum_cent[k*D + d] * inv;
        }
    }
}
