//
// Created by johan on 17/05/2025.
//
#include "kmeans.hpp"
#include <algorithm>
#include <cmath>
#include <limits>

KMeans::KMeans(const KMeansConfig& cfg)
  : cfg_(cfg),
    centroids_(cfg.K * cfg.D),
    counts_(cfg.K),
    gen_(cfg.seed)
{}

void KMeans::init_centroids() {
    std::uniform_real_distribution<> dist(0.0, 1.0);
    for (auto& c : centroids_) {
        c = dist(gen_);
    }
}

void KMeans::run(const double* data, int n) {
    int K = cfg_.K;
    int D = cfg_.D;
    int B = std::min(cfg_.batch_size, n);
    std::uniform_int_distribution<> pick(0, n - 1);

    std::vector<double> sums(size_t(K) * D);

    for (int iter = 0; iter < cfg_.local_iters; ++iter) {
        std::fill(sums.begin(), sums.end(), 0.0);
        std::fill(counts_.begin(), counts_.end(), 0);

        for (int i = 0; i < B; ++i) {
            int idx = pick(gen_);
            const double* x = data + size_t(idx) * D;

            // find nearest centroid
            int best_k = 0;
            double best_d2 = std::numeric_limits<double>::infinity();
            for (int k = 0; k < K; ++k) {
                double d2 = 0.0;
                const double* cptr = &centroids_[size_t(k) * D];
                for (int d = 0; d < D; ++d) {
                    double diff = x[d] - cptr[d];
                    d2 += diff * diff;
                }
                if (d2 < best_d2) {
                    best_d2 = d2;
                    best_k = k;
                }
            }

            // accumulate sums and counts
            double* sptr = &sums[size_t(best_k) * D];
            for (int d = 0; d < D; ++d) {
                sptr[d] += x[d];
            }
            counts_[best_k]++;
        }

        // update centroids
        for (int k = 0; k < K; ++k) {
            if (counts_[k] > 0) {
                double inv = 1.0 / counts_[k];
                double* cptr = &centroids_[size_t(k) * D];
                double* sptr = &sums[size_t(k) * D];
                for (int d = 0; d < D; ++d) {
                    cptr[d] = sptr[d] * inv;
                }
            }
        }
    }
}
