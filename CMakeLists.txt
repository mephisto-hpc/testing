CMAKE_MINIMUM_REQUIRED(VERSION 3.7.2)

PROJECT(mephisto-tests)

SET(ALPAKA_ROOT "$ENV{ALPAKA_ROOT}" CACHE STRING "The location of the alpaka library")
LIST(APPEND CMAKE_MODULE_PATH "${ALPAKA_ROOT}")
FIND_PACKAGE("alpaka" REQUIRED)

INCLUDE("${ALPAKA_ROOT}/cmake/common.cmake")
#INCLUDE("${ALPAKA_ROOT}/cmake/dev.cmake")

ALPAKA_ADD_EXECUTABLE(
    alpaka-vxv
    "alpaka-vxv.cpp")
TARGET_COMPILE_OPTIONS(
    alpaka-vxv
    PRIVATE ${ALPAKA_DEV_COMPILE_OPTIONS})
TARGET_LINK_LIBRARIES(
    alpaka-vxv
    PUBLIC "alpaka")
