//
// Created by johan on 17/05/2025.
//
#include "kmeans.hpp"
#include <algorithm>
#include <cmath>
#include <fstream>
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

bool KMeans::save(const std::string& filename) const
{
    if (centroids_.empty()) return false;

    std::ofstream out(filename, std::ios::binary);
    if (!out) return false;

    int32_t K32 = static_cast<int32_t>(cfg_.K);
    int32_t D32 = static_cast<int32_t>(cfg_.D);

    out.write(reinterpret_cast<char*>(&K32), sizeof(K32));
    out.write(reinterpret_cast<char*>(&D32), sizeof(D32));
    out.write(reinterpret_cast<const char*>(centroids_.data()),
              sizeof(double) * centroids_.size());

    return static_cast<bool>(out);
}

bool KMeans::load(const std::string& filename)
{
    std::ifstream in(filename, std::ios::binary);
    if (!in) return false;

    int32_t K32 = 0, D32 = 0;
    in.read(reinterpret_cast<char*>(&K32), sizeof(K32));
    in.read(reinterpret_cast<char*>(&D32), sizeof(D32));
    if (!in || K32 <= 0 || D32 <= 0) return false;

    cfg_.K = static_cast<int>(K32);
    cfg_.D = static_cast<int>(D32);

    centroids_.resize(size_t(cfg_.K) * cfg_.D);
    counts_.assign(cfg_.K, 0);            // reset counts

    in.read(reinterpret_cast<char*>(centroids_.data()),
            sizeof(double) * centroids_.size());

    return static_cast<bool>(in);
}
