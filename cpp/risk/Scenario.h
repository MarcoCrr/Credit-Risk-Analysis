#pragma once

#include <string>

struct Scenario {
    std::string name;

    double pd_multiplier = 1.0;
    double rho = 0.2;
    double lgd_multiplier = 1.0;
};