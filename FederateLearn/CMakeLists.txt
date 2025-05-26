cmake_minimum_required(VERSION 3.14)
project(federated_kmeans_mpi LANGUAGES C CXX)

# ---- Compiler settings -------------------------------------------------
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ---- MPI (cluster-friendly) -------------------------------------------
# Rely on the system’s module-provided Open MPI / MPICH.
find_package(MPI REQUIRED)   # supplies imported target MPI::MPI_CXX

# ---- Sources -----------------------------------------------------------
set(SRCS
    src/main.cpp
    src/kmeans.cpp
    src/dataloader.cpp
    src/mpi_helpers.cpp)

# ---- Main executable ---------------------------------------------------
add_executable(federated_kmeans ${SRCS})
target_include_directories(federated_kmeans PRIVATE include)
target_link_libraries(federated_kmeans PRIVATE MPI::MPI_CXX)

# ---- Evaluation tool ---------------------------------------------------
add_executable(eval
    src/eval.cpp
    src/kmeans.cpp
    src/dataloader.cpp
    src/mpi_helpers.cpp)
target_include_directories(eval PRIVATE include)
target_link_libraries(eval PRIVATE MPI::MPI_CXX)

# ---- Unit test ---------------------------------------------------------
enable_testing()
add_executable(kmeans_test
    tests/kmeans_sanity.cpp
    src/kmeans.cpp
    src/dataloader.cpp)
target_include_directories(kmeans_test PRIVATE include)
target_link_libraries(kmeans_test PRIVATE MPI::MPI_CXX)
add_test(NAME KMeansSanity COMMAND kmeans_test)
