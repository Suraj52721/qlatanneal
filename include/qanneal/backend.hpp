#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>

#include "qanneal/hamiltonian.hpp"

namespace qanneal {

enum class BackendKind {
    CPU,
    CUDA
};

inline BackendKind backend_from_string(std::string_view name) {
    if (name == "cpu" || name == "CPU") {
        return BackendKind::CPU;
    }
    if (name == "cuda" || name == "CUDA") {
        return BackendKind::CUDA;
    }
    throw std::invalid_argument("Unknown backend: " + std::string(name));
}

inline const char *backend_to_string(BackendKind kind) {
    switch (kind) {
    case BackendKind::CPU:
        return "cpu";
    case BackendKind::CUDA:
        return "cuda";
    }
    return "unknown";
}

class Backend {
public:
    virtual ~Backend() = default;
    virtual BackendKind kind() const = 0;
    virtual std::size_t size() const = 0;
    virtual double energy(const int8_t *spins, std::size_t n) const = 0;
    virtual double delta_energy(const int8_t *spins, std::size_t n, std::size_t flip) const = 0;
};

class CPUBackend final : public Backend {
public:
    explicit CPUBackend(std::shared_ptr<const Hamiltonian> hamiltonian)
        : ham_(std::move(hamiltonian)) {
        if (!ham_) {
            throw std::invalid_argument("CPUBackend requires a Hamiltonian.");
        }
    }

    BackendKind kind() const override { return BackendKind::CPU; }
    std::size_t size() const override { return ham_->size(); }

    double energy(const int8_t *spins, std::size_t n) const override {
        return ham_->energy(spins, n);
    }

    double delta_energy(const int8_t *spins, std::size_t n, std::size_t flip) const override {
        return ham_->delta_energy(spins, n, flip);
    }

private:
    std::shared_ptr<const Hamiltonian> ham_;
};

inline std::shared_ptr<Backend> make_backend(BackendKind kind,
                                             std::shared_ptr<const Hamiltonian> ham) {
    switch (kind) {
    case BackendKind::CPU:
        return std::make_shared<CPUBackend>(std::move(ham));
    case BackendKind::CUDA:
#if defined(QANNEAL_ENABLE_CUDA) && QANNEAL_ENABLE_CUDA
        throw std::runtime_error("CUDA backend stub: kernels not wired yet.");
#else
        throw std::runtime_error("CUDA backend not enabled at build time.");
#endif
    }
    throw std::runtime_error("Unknown backend kind.");
}

inline std::shared_ptr<Backend> make_backend(BackendKind kind, const Hamiltonian &ham) {
    auto non_owning = std::shared_ptr<const Hamiltonian>(&ham, [](const Hamiltonian *) {});
    return make_backend(kind, std::move(non_owning));
}

} // namespace qanneal
