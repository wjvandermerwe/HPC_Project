#include <iostream>
#include <vector>
#include <algorithm>

#include "mpi_helpers.hpp"
#include "dataloader.hpp"
#include "kmeans.hpp"

const int SERVER_RANK = 0;

int main(int argc, char** argv)
{
    /* ---------- 1. MPI init ---------- */
    int rank, world;
    mpi_initialize(rank, world);

    if (world < 2) {
        if (rank == 0) std::cerr << "Need ≥2 ranks (1 server + clients)\n";
        mpi_finalize();  return 1;
    }

    /* ---------- 2. Load local shard (clients only) ---------- */
    std::vector<double> data;
    int n_local = 0, local_D = 0;

    if (rank > SERVER_RANK) {
        char client_id = char('A' + (rank - 1));
        std::string fname = "./data/train_client_"; fname += client_id; fname += ".bin";

        if (!load_binary_dataset(fname, data, n_local, local_D)) {
            std::cerr << "[rank " << rank << "] cannot load " << fname << "\n";
            mpi_finalize();  return 1;
        }
        std::cout << "[rank " << rank << "] loaded " << n_local << " samples, D=" << local_D << "\n";
    }

    /* ---------- 3. Agree on global dimensionality ---------- */
    int global_D = 0;
    MPI_Allreduce(&local_D, &global_D, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    if (global_D == 0) {           // should never happen
        if (rank == 0) std::cerr << "Could not determine feature dimension!\n";
        mpi_finalize(); return 1;
    }

    /* ---------- 4. Build K-Means object ---------- */
    const int K = 10;
    KMeansConfig cfg{K, global_D, /*local_iters*/5, /*batch_size*/100, /*seed*/1234};
    KMeans km(cfg);

    /* ---------- 5. Server initialises centroids ---------- */
    if (rank == SERVER_RANK) km.init_centroids();
    broadcast_centroids(km.centroids().data(), K, global_D, SERVER_RANK);

    /* ---------- 6. Federated rounds ---------- */
    const int ROUNDS = 20;
    for (int r = 0; r < ROUNDS; ++r) {
        if (rank > SERVER_RANK)          // workers only
            km.run(data.data(), n_local);

        if (rank == SERVER_RANK) {       // aggregate on server
            server_exchange(km.centroids().data(),
                             km.counts().data(),
                             K,
                             global_D);
        } else {                         // workers send + receive
            worker_exchange(km.centroids().data(),
                            km.counts().data(),
                            K, global_D, SERVER_RANK);
        }

        if (rank == SERVER_RANK) std::cout << "[round " << r << "] done\n";
    }

    /* ---------- 7. Show a snippet of result ---------- */
    if (rank == SERVER_RANK) {
        if (km.save("centroids.bin"))
            std::cout << "Centroids saved to centroids.bin\n";
        else
            std::cerr << "Failed to save centroids!\n";
        std::cout << "Centroid[0][0..4]: ";
        for (int i = 0; i < std::min(K * global_D, 5); ++i)
            std::cout << km.centroids()[i] << ' ';
        std::cout << '\n';
    }

    mpi_finalize();
    return 0;
}
