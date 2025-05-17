//
// Created by johan on 17/05/2025.
//
#include <cassert>
#include <iostream>
#include <vector>
#include <string>
#include <filesystem>
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#include "kmeans.hpp"

int main() {
    namespace fs = std::filesystem;
    const std::string images_dir = "images";

    // Collect image paths
    std::vector<fs::path> img_files;
    for (auto& p : fs::directory_iterator(images_dir)) {
        if (p.is_regular_file()) {
            img_files.push_back(p.path());
        }
    }
    assert(!img_files.empty());
    const int n = static_cast<int>(img_files.size());

    // Load first image to get dimensions
    int width, height, channels;
    unsigned char* data0 = stbi_load(img_files[0].string().c_str(), &width, &height, &channels, 0);
    assert(data0 && "Failed to load image");
    stbi_image_free(data0);
    const int D = width * height * channels;

    // Prepare data array (n x D)
    std::vector<double> data;
    data.reserve(n * D);
    for (const auto& img_path : img_files) {
        int w, h, c;
        unsigned char* pixels = stbi_load(img_path.string().c_str(), &w, &h, &c, channels);
        assert(pixels && w == width && h == height && c == channels);

        // Normalize to [0,1]
        for (int i = 0; i < w * h * c; ++i) {
            data.push_back(pixels[i] / 255.0);
        }
        stbi_image_free(pixels);
    }

    // K-Means configuration
    KMeansConfig cfg;
    cfg.K = 2;
    cfg.D = D;
    cfg.local_iters = 1;
    cfg.batch_size = n;
    cfg.seed = 123;

    // Run K-Means
    KMeans km(cfg);
    km.init_centroids();
    auto cent_before = km.centroids();
    km.run(data.data(), n);
    auto cent_after = km.centroids();
    const auto& counts = km.counts();

    // Assertions
    assert(static_cast<int>(cent_after.size()) == cfg.K * cfg.D);
    assert(static_cast<int>(counts.size()) == cfg.K);
    int total_count = 0;
    for (int c : counts) total_count += c;
    assert(total_count == cfg.batch_size && "Sum of counts should equal batch size");

    // Centroids should change after run
    bool changed = false;
    for (size_t i = 0; i < cent_after.size(); ++i) {
        if (cent_after[i] != cent_before[i]) { changed = true; break; }
    }
    assert(changed && "Centroids did not change after run");

    std::cout << "KMeans image test passed on " << n << " images of size "
              << width << "x" << height << "x" << channels << std::endl;
    return 0;
}
