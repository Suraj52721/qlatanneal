#include <cassert>
#include <cmath>

#include "qanneal/dense_ising.hpp"
#include "qanneal/state.hpp"

int main() {
    const std::size_t n = 2;
    std::vector<double> h = {1.0, -1.0};
    std::vector<double> J = {
        0.0, 0.5,
        0.5, 0.0
    };

    qanneal::DenseIsing ham(h, J, n);

    qanneal::State s(n);
    s[0] = 1;
    s[1] = 1;

    const double e0 = ham.energy(s);
    assert(std::abs(e0 - 0.5) < 1e-12);

    const double delta = ham.delta_energy(s, 0);
    qanneal::State s2 = s;
    s2[0] = -1;
    const double e1 = ham.energy(s2);
    assert(std::abs((e1 - e0) - delta) < 1e-12);

    return 0;
}
