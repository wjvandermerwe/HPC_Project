#include <iostream>
#include <vector>

#include "mpi_helpers.hpp"
#include "dataloader.hpp"
#include "kmeans.hpp"

int main(int argc, char** argv) {
    int rank, world;
    mpi_initialize(rank, world);

    // load this rank’s data shard
    std::string filename = "data/train_client_";
    filename += char('A' + rank);  // assuming clients A, B, C, …
    filename += ".bin";

    std::vector<double> data;
    int n_samples, dim;
    if (!load_binary_dataset(filename, data, n_samples, dim)) {
        if (rank == 0)
            std::cerr << "[error] failed to load " << filename << "\n";
        mpi_finalize();
        return 1;
    }

    // configure K-Means
    int K = 10;               // number of clusters
    KMeansConfig cfg{K, dim, /*local_iters=*/5, /*batch_size=*/100, /*seed=*/1234};
    KMeans km(cfg);

    // server initializes centroids, then broadcast
    if (rank == 0) {
        km.init_centroids();
    }
    broadcast_centroids(km.centroids().data(), K, dim, /*root=*/0);

    // federated training loop
    int rounds = 20;
    for (int r = 0; r < rounds; ++r) {
        // each worker (and server) runs local mini-batch updates
        km.run(data.data(), n_samples);

        // extract local sums from centroids_ and counts_
        double * buffer = (double*)km.centroids().data(); // explicit cast stream
        if (rank == 0) {
            // server aggregates everybody’s sums→global averages into centroids
            server_aggregate(km.centroids().data(),
                              km.counts().data(),
                              buffer,
                              K, dim);
        } else {
            // workers send sums & counts, receive updated centroids
            worker_exchange(buffer,
                            km.counts().data(),
                            K, dim,
                            /*server_rank=*/0);
        }

        if (rank == 0) {
            std::cout << "[round " << r << "] done\n";
        }
    }

    // final centroids live in km.centroids() on server
    if (rank == 0) {
        std::cout << "Final centroids[0..5]: ";
        for (int i = 0; i < std::min(K * dim, 5); ++i) {
            std::cout << km.centroids()[i] << " ";
        }
        std::cout << "\n";
    }

    mpi_finalize();
    return 0;
}
