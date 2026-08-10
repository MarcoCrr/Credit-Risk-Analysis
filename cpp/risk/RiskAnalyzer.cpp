#include "RiskAnalyzer.h"

#include <algorithm>

#include "Simulator.h"

RiskReport RiskAnalyzer::analyze(
    const std::vector<Loan>& portfolio,
    const Scenario& scenario,
    int simulations)
{
    // Copy portfolio
    std::vector<Loan> stressed = portfolio;

    // Apply PD stress
    for (auto& loan : stressed)
    {
        constexpr double PD_EPSILON = 1e-6;

        loan.pd = std::clamp(
            loan.pd * scenario.pd_multiplier,
            PD_EPSILON,
            1.0 - PD_EPSILON);
    }

    // Monte Carlo
    auto losses =
        Simulator::run_correlated(
            stressed,
            simulations,
            scenario.rho);

    RiskReport report;

    report.expected_loss =
        Simulator::expected_loss(losses);

    report.var95 =
        Simulator::var(losses, 0.95);

    report.var99 =
        Simulator::var(losses, 0.99);

    return report;
}