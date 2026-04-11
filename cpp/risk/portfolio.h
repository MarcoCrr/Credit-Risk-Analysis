#pragma once
#include <vector>
#include <Eigen/Dense>
#include "Loan.h"

class Portfolio {
public:
    static std::vector<Loan> build(const Eigen::MatrixXd& X,
                                   const Eigen::VectorXd& pd_preds);
};