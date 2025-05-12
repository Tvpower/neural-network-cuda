#This is a unified version of the project for cpu and gpu optimization

cmake_minimum_required(VERSION 3.16)
project(neural_network_cuda LANGUAGES CXX CUDA) #specify both languages

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find CUDA package
find_package(CUDAToolkit REQUIRED)

#Source files
file(GLOB CPU_SOURCES "CPU/*.cpp")
file(GLOB GPU_SOURCES "GPU/*.cu")
file(GLOB UTILS_SOURCES "utils/*.cpp")
file(GLOB DATA_SOURCES "data/*.cpp")
file(GLOB TEST_SOURCES "test/*.cu") #include the test files

#include directories
include_directories(${CMAKE_SOURCE_DIR})

# --- Function to add CUDA-specific compile options ---
function(add_cuda_options target)   #This function encapsulates the CUDA-specific compile options and linking, making the Cmake code-
    if(TARGET ${target})            #more modular and readable. It also includes a check to ensure the target exist before applying CUDA options.

        target_compile_options(${target} PRIVATE
            $<$<COMPILE_LANGUAGE:CUDA>:
                -Xcompiler="-fPIC" #position Independent Code (If needed). this is neccesary when building shared libraries or in some system configurations. Might need to check this out for future implementations or idk
                -Xcudafe --display_error_number
            >
        )
        set_target_properties(${target} PROPERTIES
            CUDA_SEPARABLE_COMPILATION ON   #This enables separable compilation for CUDA code which:
                                            #Allows for separate dynamic parallelism
        )                                   #Makes linking more flexible
                                            #Speeds up compilation time for large projects
        target_link_libraries(${target} ${CUDA_LIBRARIES})
    else ()
        message(WARNING "Target '${target}' not found, CUDA options not applied.")
    endif ()
endfunction()
#Remember Nvcc is separating host code(ran in cpu) from device code (runs on gpu)

# --- Create the neural network library ---
add_library(neural_network_lib  # Changed from add_executable to add_library
        ${CPU_SOURCES}
        ${GPU_SOURCES}
        ${UTILS_SOURCES}
        ${DATA_SOURCES}
)
add_cuda_options(neural_network_lib) #Apply CUDA options to the library

# --- Create the main executable ---
add_executable(neural_network "main.cpp") # gets the main.cpp in the root directory
target_link_libraries(neural_network neural_network_lib) # Link the executable to the library

# --- Create the test executable ---
add_executable(run_tests "test/test_main.cpp" ${TEST_SOURCES})
target_link_libraries(run_tests neural_network_lib)  # Link the test executable to the library
add_cuda_options(run_tests) # Apply CUDA options to the test executable

# Enable testing
enable_testing()
add_test(NAME UnitTests COMMAND run_tests)