#pragma once

#include <stdexcept>

#include <mpi.h>

namespace qanneal::mpi {

class MPIContext {
public:
    MPIContext(int &argc, char **&argv) {
        int initialized = 0;
        MPI_Initialized(&initialized);
        if (!initialized) {
            if (MPI_Init(&argc, &argv) != MPI_SUCCESS) {
                throw std::runtime_error("MPI_Init failed.");
            }
            owns_ = true;
        }
    }

    MPIContext() {
        int initialized = 0;
        MPI_Initialized(&initialized);
        if (!initialized) {
            if (MPI_Init(nullptr, nullptr) != MPI_SUCCESS) {
                throw std::runtime_error("MPI_Init failed.");
            }
            owns_ = true;
        }
    }

    ~MPIContext() {
        int finalized = 0;
        MPI_Finalized(&finalized);
        if (owns_ && !finalized) {
            MPI_Finalize();
        }
    }

    int rank() const {
        int r = 0;
        MPI_Comm_rank(MPI_COMM_WORLD, &r);
        return r;
    }

    int size() const {
        int s = 1;
        MPI_Comm_size(MPI_COMM_WORLD, &s);
        return s;
    }

    void barrier() const { MPI_Barrier(MPI_COMM_WORLD); }

private:
    bool owns_ = false;
};

} // namespace qanneal::mpi
