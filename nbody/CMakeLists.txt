cmake_minimum_required (VERSION 3.7)

# TODO: use modern cmake?

# Enable it when you need folder organization for your targets,
# see: https://cliutils.gitlab.io/modern-cmake/chapters/features/ides.html
# set_property(GLOBAL PROPERTY USE_FOLDERS ON)

project (mephisto-nbody)

set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 14)

find_package(dash-mpi REQUIRED)
set(LIBRARIES ${LIBRARIES} ${DASH_LIBRARIES})

set(LLAMA_ROOT "${CMAKE_CURRENT_LIST_DIR}/../llama/" CACHE STRING "The location of the llama library")
list(APPEND CMAKE_PREFIX_PATH "${LLAMA_ROOT}")

find_package(llama 0.1.0 REQUIRED)
set(INCLUDE_DIRS ${INCLUDE_DIRS} ${llama_INCLUDE_DIR})
set(INCLUDE_DIRS ${INCLUDE_DIRS} ${LLAMA_ROOT}/examples/common)
add_definitions(${llama_DEFINITIONS})

set(ALPAKA_ROOT "${CMAKE_CURRENT_LIST_DIR}/../alpaka/" CACHE STRING  "The location of the alpaka library")
list(APPEND CMAKE_PREFIX_PATH "${ALPAKA_ROOT}")
find_package(alpaka REQUIRED)

set(LIBRARIES ${LIBRARIES} ${alpaka_LIBRARIES})
set(INCLUDE_DIRS ${INCLUDE_DIRS} ${alpaka_INCLUDE_DIRS})
set(DEFINITIONS ${DEFINITIONS} ${alpaka_DEFINITIONS})
#set(DEFINITIONS ${DEFINITIONS} ${ALPAKA_DEV_COMPILE_OPTIONS})

# workaround for: NVCC does not incorporate the alpaka defines
if(ALPAKA_ACC_GPU_CUDA_ENABLE OR ALPAKA_ACC_GPU_HIP_ENABLE)
    set(_ALPAKA_COMPILE_DEFINITIONS_CUDA ${_ALPAKA_COMPILE_DEFINITIONS_PUBLIC})
    list_add_prefix("-D" _ALPAKA_COMPILE_DEFINITIONS_CUDA)
    list(APPEND CUDA_NVCC_FLAGS ${_ALPAKA_COMPILE_DEFINITIONS_CUDA})
endif()

include_directories(mephisto-nbody ${INCLUDE_DIRS})
alpaka_add_executable(mephisto-nbody nbody.cpp Dummy.cpp)
target_compile_options(mephisto-nbody PRIVATE ${DEFINITIONS})
target_link_libraries(mephisto-nbody PUBLIC ${LIBRARIES})
