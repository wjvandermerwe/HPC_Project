//
// Created by johan on 17/05/2025.
//

#ifndef KMEANS_HPP
#define KMEANS_HPP

#include <vector>
#include <random>

struct KMeansConfig {
    int    K;
    int    D;
    int    local_iters;
    int    batch_size;
    unsigned seed = 42;
};

class KMeans {
public:
    explicit KMeans(const KMeansConfig& cfg);

    // initialize centroids uniformly in [0,1]
    void init_centroids();

    // run local updates on `n` samples in `data` (row-major n×D)
    void run(const double* data, int n);

    // getters
    const std::vector<double>& centroids() const { return centroids_; }
    const std::vector<int>&    counts()    const { return counts_; }

private:
    KMeansConfig         cfg_;
    std::vector<double>  centroids_;   // size K*D
    std::vector<int>     counts_;      // size K
    std::mt19937         gen_;
};

#endif // KMEANS_HPP
