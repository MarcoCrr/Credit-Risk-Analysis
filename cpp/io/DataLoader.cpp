#include "DataLoader.h"
#include <fstream>
#include <sstream>
#include <iostream>
#include <vector>

Eigen::MatrixXd DataLoader::loadFeatures(const std::string& path) {
    std::ifstream file(path);
    std::string line;

    std::vector<std::vector<double>> data;

    if (!file.is_open()) {
        throw std::runtime_error("Could not open file: " + path);
    }

    // Skip header
    std::getline(file, line);

    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string value;
        std::vector<double> row;

        while (std::getline(ss, value, ',')) {
            try {
                row.push_back(std::stod(value));
            } catch (...) {
                std::cerr << "Invalid value: '" << value << "'" << std::endl;
                row.push_back(0.0);  // fallback
            }
        }
        if (!data.empty() && row.size() != data[0].size()) {
            std::cerr << "Skipping malformed row with size "
                    << row.size() << std::endl;
            continue;
        }

        data.push_back(row);
    }


    if (data.empty()) {
        throw std::runtime_error("No data loaded from file: " + path);
    }

    int rows = data.size();
    int cols = data[0].size();

    std::cout << "Rows: " << rows << ", Cols: " << cols << std::endl;

    Eigen::MatrixXd X(rows, cols);

    for (int i = 0; i < rows; ++i)
        for (int j = 0; j < cols; ++j)
            X(i, j) = data[i][j];

    return X;
}

Eigen::VectorXd DataLoader::loadTarget(const std::string& path) {
    std::ifstream file(path);
    std::string line;

    std::vector<double> data;

    if (!file.is_open()) {
        throw std::runtime_error("Could not open file: " + path);
    }

    // Skip header
    std::getline(file, line);

    while (std::getline(file, line)) {
        data.push_back(std::stod(line));
    }

    if (data.empty()) {
        throw std::runtime_error("No target data loaded!");
    }

    Eigen::VectorXd y(data.size());

    for (int i = 0; i < data.size(); ++i)
        y(i) = data[i];

    return y;
}