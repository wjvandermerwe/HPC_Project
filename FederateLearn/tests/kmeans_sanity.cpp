// tests/kmeans_bin_test.cpp

#include <cassert>
#include <iostream>
#include <vector>

#include "dataloader.hpp"
#include "kmeans.hpp"

int main() {
    // 1) Load a binary shard (pick whichever client you like)
    std::string filename = "../tests/data/test_client_A.bin";
    std::vector<double> full_data;
    int full_n, D;
    bool ok = load_binary_dataset(filename, full_data, full_n, D);
    assert(ok && full_n > 0 && D > 0 && "Failed to load binary dataset");

    // 2) Take exactly 100 samples (or fewer if the shard is small)
    int test_n = std::min(full_n, 100);
    std::vector<double> data(test_n * D);
    std::copy_n(full_data.data(), test_n * D, data.data());

    KMeansConfig cfg;
    cfg.K           = 2;
    cfg.D           = D;
    cfg.local_iters = 1;
    cfg.batch_size  = test_n;
    cfg.seed        = 42;

    KMeans km(cfg);
    km.init_centroids();
    auto cent_before = km.centroids();
    km.run(data.data(), test_n);
    auto cent_after  = km.centroids();
    const auto& counts = km.counts();

    assert((int)cent_after.size() == cfg.K * D);
    assert((int)counts.size()    == cfg.K);

    int total_count = 0;
    for (int c : counts) {
        total_count += c;
    }
    assert(total_count == test_n && "Counts must sum to test_n");

    bool changed = false;
    for (size_t i = 0; i < cent_after.size(); ++i) {
        if (cent_after[i] != cent_before[i]) {
            changed = true;
            break;
        }
    }
    assert(changed && "Centroids did not change after run()");

    std::cout << "kmeans_bin_test passed: ran on "
              << test_n << " samples of dimension "
              << D << "\n";
    return 0;
}
