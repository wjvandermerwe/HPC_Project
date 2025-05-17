cmake_minimum_required(VERSION 3.14)
project(federated_kmeans_mpi LANGUAGES C CXX)
set(MPI_HOME "C:/Program Files (x86)/Microsoft SDKs/MPI")

set(MPI_C_INCLUDE_PATH   "${MPI_HOME}/Include")
set(MPI_CXX_INCLUDE_PATH "${MPI_HOME}/Include")

set(MPI_C_LIBRARIES      "${MPI_HOME}/Lib/x64/msmpi.lib")
set(MPI_CXX_LIBRARIES    "${MPI_HOME}/Lib/x64/msmpi.lib")
# --- Compiler settings -------------------------------------------------
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# --- Dependencies ------------------------------------------------------
find_package(MPI REQUIRED)        # provides imported target MPI::MPI_CXXS

# --- Targets -----------------------------------------------------------
set(SOURCES
        src/main.cpp
        src/kmeans.cpp
        src/dataloader.cpp
        src/mpi_helpers.cpp
)

add_executable(federated_kmeans ${SOURCES})
target_include_directories(federated_kmeans PRIVATE include)
target_link_libraries(federated_kmeans PRIVATE MPI::MPI_CXX)


# --- Unit test ---------------------------------------------------------
enable_testing()
add_executable(kmeans_test tests/kmeans_sanity.cpp src/kmeans.cpp)
target_include_directories(kmeans_test PRIVATE include)
target_link_libraries(kmeans_test PRIVATE MPI::MPI_CXX)
add_test(NAME KMeansSanity COMMAND kmeans_test)