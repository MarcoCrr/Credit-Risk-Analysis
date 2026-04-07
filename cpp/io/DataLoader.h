#pragma once
#include <Eigen/Dense>
#include <string>

class DataLoader {
public:
    static Eigen::MatrixXd loadFeatures(const std::string& path);
    static Eigen::VectorXd loadTarget(const std::string& path);
};