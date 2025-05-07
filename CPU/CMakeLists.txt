cmake_minimum_required(VERSION 3.15)
project(neural_network_cpu LANGUAGES CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Include directories
include_directories(${CMAKE_SOURCE_DIR})
include_directories(${CMAKE_SOURCE_DIR}/..)

# Add all CPU source files
file(GLOB CPU_SOURCES "CPU/*.cpp")


# Add utility functions
file(GLOB UTILS_SOURCES "utils/*.cpp")
file(GLOB DATA_SOURCES "data/*.cpp")

# Create executable
add_executable(neural_network_cpu
        ${CPU_SOURCES}
        ${UTILS_SOURCES}
        ${DATA_SOURCES}
)

# Add compiler options if needed
target_compile_options(neural_network_cpu PRIVATE
        -Wall
        -Wextra
)