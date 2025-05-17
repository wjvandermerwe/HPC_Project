//
// Created by johan on 17/05/2025.
//

#ifndef DATALOADER_HPP
#define DATALOADER_HPP

#include <string>
#include <vector>

// Loads a dataset from `filename`.
// On success returns true, and sets:
//   - data to a vector of length n*D (row-major doubles),
//   - n to the number of samples,
//   - D to the feature dimension.
bool load_binary_dataset(const std::string& filename,
                         std::vector<double>& data,
                         int& n,
                         int& D);

#endif // DATALOADER_HPP
