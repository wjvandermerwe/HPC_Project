//
// Created by johan on 18/05/2025.
//
#include <iostream>
#include <vector>
#include <limits>
#include <numeric>

#include "mpi_helpers.hpp"
#include "dataloader.hpp"
#include "kmeans.hpp"

// -------- helper to get nearest cluster and distance^2 ------------
static inline std::pair<int,double>
nearest(const double* x, const std::vector<double>& cent, int K, int D)
{
    int best_k = 0;
    double best_d2 = std::numeric_limits<double>::infinity();
    for (int k = 0; k < K; ++k) {
        const double* c = &cent[k*D];
        double d2 = 0.0;
        for (int d = 0; d < D; ++d) {
            double diff = x[d] - c[d];
            d2 += diff*diff;
        }
        if (d2 < best_d2) { best_d2 = d2; best_k = k; }
    }
    return {best_k, best_d2};
}
// ------------------------------------------------------------------

int main(int argc, char** argv)
{
    int rank, world;
    mpi_initialize(rank, world);

    /* ---------- 1. Load global centroids on server, then broadcast --------- */
    KMeansConfig dummy{0,0,0,0};      // cfg_ will be filled by load()
    KMeans km(dummy);
    if (rank == 0) {
        if (!km.load("centroids.bin")) {
            std::cerr << "Server could not load centroids.bin\n";
            mpi_finalize();  return 1;
        }
    }
    // Broadcast K & D so clients can size buffers
    int meta[2] = { km.cfg().K, km.cfg().D };
    MPI_Bcast(meta, 2, MPI_INT, 0, MPI_COMM_WORLD);
    if (rank != 0) {
        km = KMeans({meta[0], meta[1], 0, 0});   // construct with correct dims
        km.centroids().resize(size_t(meta[0])*meta[1]);
    }
    broadcast_centroids(km.centroids().data(), meta[0], meta[1], 0);

    /* ---------- 2. Load this rank’s test shard (clients only) -------------- */
    std::vector<double> data;
    int n=0, D=meta[1];
    if (rank > 0) {
        char id = char('A'+(rank-1));
        std::string fname = "./data/test_client_"; fname+=id; fname+=".bin";
        if (!load_binary_dataset(fname, data, n, D)) {
            std::cerr << "[rank " << rank << "] failed to load " << fname << "\n";
            mpi_finalize(); return 1;
        }
    }

    /* ---------- 3. Local evaluation --------------------------------------- */
    double local_inertia = 0.0;
    std::vector<int> cluster_cnt(meta[0], 0);

    for (int i = 0; i < n; ++i) {
        const double* x = &data[i*D];
        auto [k, d2]  = nearest(x, km.centroids(), meta[0], D);
        local_inertia += d2;
        cluster_cnt[k]++;
    }

    /* ---------- 4. Reduce to server --------------------------------------- */
    double global_inertia = 0.0;
    std::vector<int> global_cnt(meta[0]);

    MPI_Reduce(&local_inertia, &global_inertia, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(cluster_cnt.data(), global_cnt.data(),
               meta[0], MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    /* ---------- 5. Print results on server -------------------------------- */
    if (rank == 0) {
        int total = std::accumulate(global_cnt.begin(), global_cnt.end(), 0);
        double avg_dist = (total ? std::sqrt(global_inertia / total) : 0.0);

        std::cout << "=== Federated K-Means Evaluation ===\n";
        std::cout << "K        : " << meta[0] << "\n";
        std::cout << "D        : " << D << "\n";
        std::cout << "Samples  : " << total << "\n";
        std::cout << "Inertia  : " << global_inertia << "\n";
        std::cout << "Avg dist : " << avg_dist << "\n";
        std::cout << "Counts per cluster:\n";
        for (int k = 0; k < meta[0]; ++k)
            std::cout << "  C" << k << ": " << global_cnt[k] << "\n";
    }

    mpi_finalize();
    return 0;
}
