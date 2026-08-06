#pragma once

#include <vector>

#include "Loan.h"
#include "Scenario.h"
#include "RiskReport.h"

class RiskAnalyzer {

public:

    static RiskReport analyze(
        const std::vector<Loan>& portfolio,
        const Scenario& scenario,
        int simulations = 1000);

};