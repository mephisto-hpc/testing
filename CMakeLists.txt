CMAKE_MINIMUM_REQUIRED(VERSION 3.6.0)

PROJECT(mephisto-tests)

SET(CMAKE_CXX_STANDARD 11)
SET(CMAKE_CXX_STANDARD_REQUIRED ON)

SET(ALPAKA_ROOT "${CMAKE_CURRENT_LIST_DIR}/alpaka" CACHE STRING "The location of the alpaka library")
LIST(APPEND CMAKE_MODULE_PATH "${ALPAKA_ROOT}")
FIND_PACKAGE("alpaka" REQUIRED)

INCLUDE("${ALPAKA_ROOT}/cmake/common.cmake")
#INCLUDE("${ALPAKA_ROOT}/cmake/dev.cmake")

IF(CMAKE_VERSION VERSION_LESS 3.7.0)
    INCLUDE_DIRECTORIES(
        ${alpaka_INCLUDE_DIRS})
    ADD_DEFINITIONS(
        ${alpaka_DEFINITIONS})
ENDIF()

ALPAKA_ADD_EXECUTABLE(
    alpaka-mxv-cpu
    "alpaka-mxv-cpu.cpp")
TARGET_LINK_LIBRARIES(
    alpaka-mxv-cpu
    PUBLIC "alpaka")

IF(ALPAKA_ACC_GPU_CUDA_ENABLE)
    ALPAKA_ADD_EXECUTABLE(
        alpaka-mxv-gpu
        "alpaka-mxv-gpu.cpp")
    TARGET_LINK_LIBRARIES(
        alpaka-mxv-gpu
        PUBLIC "alpaka")
ENDIF()

FIND_PACKAGE(MPI)

IF(MPI_CXX_FOUND)

    FIND_PACKAGE(BLAS)
    IF(BLAS_FOUND)
        FIND_PATH(CBLAS_INCLUDE_PATH
            NAMES cblas.h
            DOC "The path to CBLAS header files")
        IF(${CBLAS_INCLUDE_PATH} STREQUAL "")
            set(BLAS_FOUND OFF CACHE BOOL "BLAS_FOUND" FORCE)
        ELSE()
            MESSAGE(STATUS "BLAS lib:      ${BLAS_LIBRARIES}")
            MESSAGE(STATUS "CBLAS include: ${CBLAS_INCLUDE_PATH}")
            ADD_DEFINITIONS("-DHAVE_CBLAS")
        ENDIF()
    ENDIF()

    FIND_PATH(DASH_INCLUDE_PATH
        NAMES libdash.h
        DOC "The path to DASH header files")

    FIND_LIBRARY(DASH_MPI_LIBRARY NAMES dash-mpi)
    FIND_LIBRARY(DART_MPI_LIBRARY NAMES dart-mpi)
    FIND_LIBRARY(DART_BASE_LIBRARY NAMES dart-base)

    MESSAGE(STATUS "DASH: ${DASH_INCLUDE_PATH}")
    MESSAGE(STATUS "DASH: ${DASH_MPI_LIBRARY} ${DART_MPI_LIBRARY} ${DART_BASE_LIBRARY}")

    INCLUDE_DIRECTORIES(
        "${MPI_CXX_INCLUDE_PATH};${DASH_INCLUDE_PATH};${CBLAS_INCLUDE_PATH}")

    ADD_EXECUTABLE(
        dash-mxv
        "dash-mxv.cpp")
    TARGET_LINK_LIBRARIES(
        dash-mxv
        "${DASH_MPI_LIBRARY};${DART_MPI_LIBRARY};${DART_BASE_LIBRARY};${BLAS_LIBRARIES};${MPI_CXX_LIBRARIES}")

    ALPAKA_ADD_EXECUTABLE(
        dash-alpaka-mxv-cpu
        "dash-alpaka-mxv-cpu.cpp")
    TARGET_LINK_LIBRARIES(
        dash-alpaka-mxv-cpu
        PUBLIC "alpaka;${DASH_MPI_LIBRARY};${DART_MPI_LIBRARY};${DART_BASE_LIBRARY};${MPI_CXX_LIBRARIES}")

    IF(ALPAKA_ACC_GPU_CUDA_ENABLE)
        ALPAKA_ADD_EXECUTABLE(
            dash-alpaka-mxv-gpu
            "dash-alpaka-mxv-gpu.cpp")
        TARGET_LINK_LIBRARIES(
            dash-alpaka-mxv-gpu
            PUBLIC "alpaka;${DASH_MPI_LIBRARY};${DART_MPI_LIBRARY};${DART_BASE_LIBRARY};${MPI_CXX_LIBRARIES}")
    ENDIF()

ENDIF()
