cmake_minimum_required(VERSION 3.31)
project(neural_network_cuda LANGUAGES CXX CUDA)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set CUDA options
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES "120")  # Targeting recent CUDA architectures

# Find CUDA package
find_package(CUDAToolkit REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})

# Include directories
include_directories(${CMAKE_SOURCE_DIR})
include_directories(${CMAKE_SOURCE_DIR}/..)

# Add all .cu files
file(GLOB GPU_SOURCES "*.cu")
# add utility functions
file(GLOB UTILS_SOURCES "../utils/*.cpp")
file(GLOB DATA_SOURCES "../data/*.cpp")


# Create executable
add_executable(neural_network_gpu
        ${GPU_SOURCES}
        ${UTILS_SOURCES}
        ${DATA_SOURCES}
)

# Set CUDA specific compile options
set_target_properties(neural_network_gpu PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
)

# Link against CUDA libraries
target_link_libraries(neural_network_gpu ${CUDA_LIBRARIES})

# Add additional compile options if needed
target_compile_options(neural_network_gpu PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
        -Xcudafe --display_error_number
        >)
