#include "StressTester.h"

#include <algorithm>

std::vector<Loan>
StressTester::shockPD(const std::vector<Loan>& portfolio,
                      double multiplier)
{
    std::vector<Loan> stressed = portfolio;

    for (auto& loan : stressed)
    {
        loan.pd = std::min(loan.pd * multiplier, 1.0);
    }

    return stressed;
}