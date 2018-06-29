cmake_minimum_required (VERSION 3.3)

SET_PROPERTY(GLOBAL PROPERTY USE_FOLDERS ON)

project (mephisto-nbody)

set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 11)

SET(LLAMA_ROOT "${CMAKE_CURRENT_LIST_DIR}/../llama/" CACHE STRING "The location of the llama library")
LIST(APPEND CMAKE_PREFIX_PATH "${LLAMA_ROOT}")

find_package(llama 0.1.0 REQUIRED)
set(INCLUDE_DIRS ${INCLUDE_DIRS} ${llama_INCLUDE_DIR})
set(INCLUDE_DIRS ${INCLUDE_DIRS} ${LLAMA_ROOT}/examples/common)
add_definitions(${llama_DEFINITIONS})

set(ALPAKA_ROOT "${CMAKE_CURRENT_LIST_DIR}/../alpaka/" CACHE STRING  "The location of the alpaka library")
LIST(APPEND CMAKE_PREFIX_PATH "${ALPAKA_ROOT}")
find_package(alpaka REQUIRED)

set(LIBRARIES ${LIBRARIES} ${alpaka_LIBRARIES})
set(INCLUDE_DIRS ${INCLUDE_DIRS} ${alpaka_INCLUDE_DIRS})
set(DEFINITIONS ${DEFINITIONS} ${alpaka_DEFINITIONS})
#set(DEFINITIONS ${DEFINITIONS} ${ALPAKA_DEV_COMPILE_OPTIONS})

FIND_PACKAGE(dash-mpi REQUIRED)
set(LIBRARIES ${LIBRARIES} ${DASH_LIBRARIES})

include_directories(mephisto-nbody ${INCLUDE_DIRS})
ALPAKA_ADD_EXECUTABLE(mephisto-nbody "nbody.cpp;Dummy.cpp")
target_compile_options(mephisto-nbody PRIVATE ${DEFINITIONS})
target_link_libraries(mephisto-nbody PUBLIC ${LIBRARIES})
