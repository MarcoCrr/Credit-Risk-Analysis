#pragma once
#include <vector>
#include "Loan.h"

class Simulator {
public:
    static double simulate_once(const std::vector<Loan>& portfolio);

    static std::vector<double> run(const std::vector<Loan>& portfolio,
                                   int n_simulations);

    static double expected_loss(const std::vector<double>& losses);

    static double var(std::vector<double> losses, double alpha);

    static double simulate_once_correlated(const std::vector<Loan>& portfolio,
                                       double rho);

    static std::vector<double> run_correlated(const std::vector<Loan>& portfolio,
                                            int n_simulations,
                                            double rho);
};