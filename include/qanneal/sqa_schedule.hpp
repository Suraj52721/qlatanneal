#pragma once

#include <cstddef>
#include <stdexcept>
#include <vector>

namespace qanneal {

struct SQASchedule {
    std::vector<double> betas;
    std::vector<double> gammas;

    std::size_t size() const { return betas.size(); }

    static SQASchedule from_vectors(std::vector<double> betas_in,
                                    std::vector<double> gammas_in) {
        if (betas_in.empty()) {
            throw std::invalid_argument("SQA schedule must contain betas.");
        }
        if (betas_in.size() != gammas_in.size()) {
            throw std::invalid_argument("SQA schedule betas/gammas length mismatch.");
        }
        SQASchedule sched;
        sched.betas = std::move(betas_in);
        sched.gammas = std::move(gammas_in);
        return sched;
    }
};

}
