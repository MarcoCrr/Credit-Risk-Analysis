#include "Simulator.h"
#include <random>
#include <algorithm>

double Simulator::simulate_once(const std::vector<Loan>& portfolio) {

    static std::mt19937 gen(42);
    std::uniform_real_distribution<> dis(0.0, 1.0);

    double total_loss = 0.0;

    for (const auto& loan : portfolio) {
        double u = dis(gen);

        if (u < loan.pd) {
            total_loss += loan.ead * loan.lgd;
        }
    }

    return total_loss;
}

std::vector<double> Simulator::run(const std::vector<Loan>& portfolio,
                                   int n_simulations) {

    std::vector<double> losses;

    for (int i = 0; i < n_simulations; ++i) {
        losses.push_back(simulate_once(portfolio));
    }

    return losses;
}

double Simulator::expected_loss(const std::vector<double>& losses) {
    double sum = 0.0;
    for (double l : losses) sum += l;
    return sum / losses.size();
}

double Simulator::var(std::vector<double> losses, double alpha) {
    std::sort(losses.begin(), losses.end());

    int index = static_cast<int>(alpha * losses.size());
    return losses[index];
}