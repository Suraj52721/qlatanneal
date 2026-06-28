cmake_minimum_required(VERSION 3.18)

project(qanneal VERSION 0.4.0 LANGUAGES CXX)

option(QANNEAL_ENABLE_OPENMP "Enable OpenMP" ON)
option(QANNEAL_ENABLE_CUDA "Enable CUDA backend" OFF)
option(QANNEAL_ENABLE_MPI "Enable MPI support" OFF)
option(QANNEAL_BUILD_TESTS "Build tests" ON)
option(QANNEAL_BUILD_PYTHON "Build Python bindings" OFF)

# Apply C++17 to all targets (tests included).
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# When building via scikit-build-core (pip), always enable Python bindings.
if(DEFINED SKBUILD)
    set(QANNEAL_BUILD_PYTHON ON CACHE BOOL "Build Python bindings" FORCE)
endif()

add_library(qanneal_core
    src/annealer.cpp
    src/ctpimc_annealer.cpp
    src/dense_ising.cpp
    src/sparse_ising.cpp
    src/sqa_annealer.cpp
    src/sqa_parallel_tempering.cpp
    src/replica_annealer.cpp
    src/parallel_tempering.cpp
    src/qubo.cpp
    src/third_party/dwave_pimc/localPIMC.cpp
)

add_library(qanneal::core ALIAS qanneal_core)

set_target_properties(qanneal_core PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS NO
    POSITION_INDEPENDENT_CODE ON
)

target_include_directories(qanneal_core
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src/third_party/dwave_pimc
)

if(QANNEAL_ENABLE_CUDA)
    target_compile_definitions(qanneal_core PUBLIC QANNEAL_ENABLE_CUDA=1)
else()
    target_compile_definitions(qanneal_core PUBLIC QANNEAL_ENABLE_CUDA=0)
endif()

if(QANNEAL_ENABLE_OPENMP)
    find_package(OpenMP)
    if(OpenMP_CXX_FOUND)
        target_link_libraries(qanneal_core PUBLIC OpenMP::OpenMP_CXX)
    endif()
endif()

# Enable native SIMD (AVX2 on x86, NEON/SVE on ARM/Apple-M).
# Guarded to MSVC which uses /arch:AVX2 instead, and to compilers that
# understand -march/-mcpu.
if(NOT MSVC)
    include(CheckCXXCompilerFlag)
    check_cxx_compiler_flag("-march=native" _QANNEAL_HAS_MARCH_NATIVE)
    if(_QANNEAL_HAS_MARCH_NATIVE)
        target_compile_options(qanneal_core PRIVATE -march=native)
        message(STATUS "qanneal: -march=native enabled (SIMD auto-vectorization)")
    else()
        # Apple Clang on ARM may not recognise -march=native; try -mcpu=native
        check_cxx_compiler_flag("-mcpu=native" _QANNEAL_HAS_MCPU_NATIVE)
        if(_QANNEAL_HAS_MCPU_NATIVE)
            target_compile_options(qanneal_core PRIVATE -mcpu=native)
            message(STATUS "qanneal: -mcpu=native enabled (SIMD auto-vectorization)")
        endif()
    endif()
endif()

if(QANNEAL_ENABLE_CUDA)
    enable_language(CUDA)
    message(STATUS "CUDA backend enabled (stub; kernels not yet wired)")
endif()

if(QANNEAL_ENABLE_MPI)
    set(QANNEAL_MPI_HOME "" CACHE PATH "Root of MPI (OpenMPI) installation")
    if(NOT QANNEAL_MPI_HOME AND DEFINED ENV{MPI_HOME})
        set(QANNEAL_MPI_HOME "$ENV{MPI_HOME}")
    endif()
    if(NOT QANNEAL_MPI_HOME AND DEFINED ENV{OPENMPI_HOME})
        set(QANNEAL_MPI_HOME "$ENV{OPENMPI_HOME}")
    endif()
    if(QANNEAL_MPI_HOME)
        list(PREPEND CMAKE_PREFIX_PATH "${QANNEAL_MPI_HOME}")
        if(NOT MPI_CXX_COMPILER)
            set(MPI_CXX_COMPILER "${QANNEAL_MPI_HOME}/bin/mpicxx" CACHE FILEPATH "MPI CXX compiler")
        endif()
    endif()

    find_package(MPI REQUIRED)
    if(DEFINED MPI_CXX_VENDOR)
        if(NOT MPI_CXX_VENDOR MATCHES "Open MPI|OpenMPI|Open MPI")
            message(WARNING "MPI vendor '${MPI_CXX_VENDOR}' detected; OpenMPI is recommended.")
        else()
            message(STATUS "MPI vendor: ${MPI_CXX_VENDOR}")
        endif()
    else()
        message(STATUS "MPI vendor not reported. If using OpenMPI, set QANNEAL_MPI_HOME or MPI_CXX_COMPILER.")
    endif()

    add_library(qanneal_mpi
        src/mpi_replica.cpp
    )
    target_link_libraries(qanneal_mpi PUBLIC qanneal_core MPI::MPI_CXX)
    target_include_directories(qanneal_mpi
        PUBLIC
            $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
            $<INSTALL_INTERFACE:include>
    )
    target_compile_definitions(qanneal_mpi PUBLIC QANNEAL_ENABLE_MPI=1)

    add_executable(qanneal_mpi_example examples/mpi/sa_mpi.cpp)
    target_link_libraries(qanneal_mpi_example PRIVATE qanneal_mpi)
endif()

if(QANNEAL_BUILD_PYTHON)
    set(PYBIND11_FINDPYTHON ON)
    find_package(pybind11 CONFIG REQUIRED)
    pybind11_add_module(_qanneal python/bindings.cpp)
    target_link_libraries(_qanneal PRIVATE qanneal_core)
    if(NOT SKBUILD)
        set_target_properties(_qanneal PROPERTIES
            LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/python/qanneal"
            RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/python/qanneal"
        )
    endif()
    install(TARGETS _qanneal
        LIBRARY DESTINATION qanneal
        RUNTIME DESTINATION qanneal
        ARCHIVE DESTINATION qanneal
    )
endif()

if(QANNEAL_BUILD_TESTS)
    include(CTest)
    add_executable(qanneal_tests tests/test_dense_ising.cpp)
    target_link_libraries(qanneal_tests PRIVATE qanneal_core)
    add_test(NAME qanneal_tests COMMAND qanneal_tests)

    add_executable(qanneal_sqa_tests tests/test_sqa_basic.cpp)
    target_link_libraries(qanneal_sqa_tests PRIVATE qanneal_core)
    add_test(NAME qanneal_sqa_tests COMMAND qanneal_sqa_tests)

    add_executable(qanneal_replica_tests tests/test_replica_annealer.cpp)
    target_link_libraries(qanneal_replica_tests PRIVATE qanneal_core)
    add_test(NAME qanneal_replica_tests COMMAND qanneal_replica_tests)

    add_executable(qanneal_pt_tests tests/test_parallel_tempering.cpp)
    target_link_libraries(qanneal_pt_tests PRIVATE qanneal_core)
    add_test(NAME qanneal_pt_tests COMMAND qanneal_pt_tests)

    add_executable(qanneal_sqapt_tests tests/test_sqa_parallel_tempering.cpp)
    target_link_libraries(qanneal_sqapt_tests PRIVATE qanneal_core)
    add_test(NAME qanneal_sqapt_tests COMMAND qanneal_sqapt_tests)
endif()

install(TARGETS qanneal_core EXPORT qannealTargets
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin
)

if(QANNEAL_ENABLE_MPI)
    install(TARGETS qanneal_mpi EXPORT qannealTargets
        ARCHIVE DESTINATION lib
        LIBRARY DESTINATION lib
        RUNTIME DESTINATION bin
    )
endif()

install(DIRECTORY include/ DESTINATION include)

include(CMakePackageConfigHelpers)
write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/QAnnealConfigVersion.cmake
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)

configure_package_config_file(
    ${CMAKE_CURRENT_LIST_DIR}/cmake/QAnnealConfig.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/QAnnealConfig.cmake
    INSTALL_DESTINATION lib/cmake/QAnneal
)

install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/QAnnealConfig.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/QAnnealConfigVersion.cmake
    DESTINATION lib/cmake/QAnneal
)

install(EXPORT qannealTargets
    NAMESPACE qanneal::
    DESTINATION lib/cmake/QAnneal
)
