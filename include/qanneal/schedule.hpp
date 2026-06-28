#pragma once

#include <cstddef>
#include <stdexcept>
#include <vector>

namespace qanneal {

struct AnnealSchedule {
    std::vector<double> betas;

    std::size_t size() const { return betas.size(); }

    static AnnealSchedule linear(double beta_start, double beta_end, std::size_t steps) {
        if (steps == 0) {
            throw std::invalid_argument("Schedule steps must be > 0.");
        }
        AnnealSchedule schedule;
        schedule.betas.reserve(steps);
        if (steps == 1) {
            schedule.betas.push_back(beta_end);
            return schedule;
        }
        const double step = (beta_end - beta_start) / static_cast<double>(steps - 1);
        for (std::size_t i = 0; i < steps; ++i) {
            schedule.betas.push_back(beta_start + step * static_cast<double>(i));
        }
        return schedule;
    }
};

}
