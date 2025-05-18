//
// Created by johan on 17/05/2025.
//
// src/data_loader.cpp
#include "dataloader.hpp"
#include <fstream>
#include <cstdint>

bool load_binary_dataset(const std::string& filename,
                         std::vector<double>& data,
                         int& n,
                         int& D)
{
    std::ifstream in(filename, std::ios::binary);
    if (!in) return false;

    // read header
    int32_t n32 = 0, D32 = 0;
    in.read(reinterpret_cast<char*>(&n32), sizeof(n32));
    in.read(reinterpret_cast<char*>(&D32), sizeof(D32));
    if (!in || n32 <= 0 || D32 <= 0) return false;

    n = static_cast<int>(n32);
    D = static_cast<int>(D32);

    // allocate and read payload
    data.resize(static_cast<size_t>(n) * D);
    in.read(reinterpret_cast<char*>(data.data()),
            sizeof(double) * data.size());
    if (!in) return false;

    return true;
}
