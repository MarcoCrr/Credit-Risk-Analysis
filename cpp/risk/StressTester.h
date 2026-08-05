#pragma once

#include <vector>
#include "Loan.h"

class StressTester {
public:
    static std::vector<Loan>
    shockPD(const std::vector<Loan>& portfolio,
            double multiplier);
};