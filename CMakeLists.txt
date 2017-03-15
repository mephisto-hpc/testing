CMAKE_MINIMUM_REQUIRED(VERSION 3.6.0)

PROJECT(mephisto-tests)

SET(ALPAKA_ROOT "${CMAKE_CURRENT_LIST_DIR}/alpaka" CACHE STRING "The location of the alpaka library")
LIST(APPEND CMAKE_MODULE_PATH "${ALPAKA_ROOT}")
FIND_PACKAGE("alpaka" REQUIRED)

INCLUDE("${ALPAKA_ROOT}/cmake/common.cmake")
INCLUDE("${ALPAKA_ROOT}/cmake/dev.cmake")

IF(CMAKE_VERSION VERSION_LESS 3.7.0)
    INCLUDE_DIRECTORIES(
        ${alpaka_INCLUDE_DIRS})
    ADD_DEFINITIONS(
        ${alpaka_DEFINITIONS})
ENDIF()
ALPAKA_ADD_EXECUTABLE(
    alpaka-mxv
    "alpaka-mxv.cpp")
TARGET_LINK_LIBRARIES(
    alpaka-mxv
    PUBLIC "alpaka")
