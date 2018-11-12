cmake_minimum_required(VERSION 3.7)

PROJECT(tests)

SET(CMAKE_CXX_STANDARD 14)
SET(CMAKE_CXX_STANDARD_REQUIRED ON)

SET(ALPAKA_ROOT "${CMAKE_CURRENT_LIST_DIR}/../../alpaka" CACHE STRING "The location of the alpaka library")
LIST(APPEND CMAKE_MODULE_PATH "${ALPAKA_ROOT}")

FIND_PACKAGE("alpaka" REQUIRED)
FIND_PACKAGE(DASH-MPI)

INCLUDE("${ALPAKA_ROOT}/cmake/common.cmake")
#INCLUDE("${ALPAKA_ROOT}/cmake/dev.cmake")

ADD_DEFINITIONS("-D_GLIBCXX_USE_C99=1")
INCLUDE_DIRECTORIES("${gtest_SOURCE_DIR}/include")
INCLUDE_DIRECTORIES(
    ${CMAKE_CURRENT_LIST_DIR}/../include)
IF(CMAKE_VERSION VERSION_LESS 3.7.0)
    INCLUDE_DIRECTORIES(
        ${alpaka_INCLUDE_DIRS})
    ADD_DEFINITIONS(
        ${alpaka_DEFINITIONS})
ENDIF()

ALPAKA_ADD_EXECUTABLE(
    array_test
    "array_test.cpp")
TARGET_LINK_LIBRARIES(
    array_test
    PUBLIC "alpaka;gtest_main")

IF(DASH-MPI_FOUND)
    ALPAKA_ADD_EXECUTABLE(
        for_each_test
        "for_each_test.cpp")
    TARGET_LINK_LIBRARIES(
        for_each_test
        PUBLIC "alpaka;${DASH_LIBRARIES};gtest_main")
ENDIF()
