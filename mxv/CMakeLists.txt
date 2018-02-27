CMAKE_MINIMUM_REQUIRED(VERSION 3.6.0)

PROJECT(mephisto-mxv)

SET(CMAKE_CXX_STANDARD 11)
SET(CMAKE_CXX_STANDARD_REQUIRED ON)

SET(ALPAKA_ROOT "${CMAKE_CURRENT_LIST_DIR}/../alpaka" CACHE STRING "The location of the alpaka library")
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

FIND_PACKAGE(DASH-MPI)
IF(DASH-MPI_FOUND)
    MESSAGE(STATUS "Found DASH: ${DASH_INSTALL_PREFIX} (found version ${DASH_VERSION_MAJOR}.${DASH_VERSION_MINOR}.${DASH_VERSION_PATCH})")

    FIND_PACKAGE(BLAS)
    IF(BLAS_FOUND AND "${BLAS_DEFINE}" STREQUAL "")
        FIND_PATH(CBLAS_INCLUDE_PATH
            NAMES cblas.h
            DOC "The path to CBLAS header files")
        IF("${CBLAS_INCLUDE_PATH}" STREQUAL "CBLAS_INCLUDE_PATH-NOTFOUND")
            unset(CBLAS_INCLUDE_PATH)
            FIND_PATH(CBLAS_INCLUDE_PATH
                NAMES mkl_cblas.h
                HINTS $ENV{MKL_INC}
                DOC "The path to CBLAS header files")
            IF(NOT "${CBLAS_INCLUDE_PATH}" STREQUAL "CBLAS_INCLUDE_PATH-NOTFOUND")
                SET(BLAS_DEFINE "HAVE_MKL_CBLAS")
            ENDIF()
        ELSE()
            SET(BLAS_DEFINE "HAVE_CBLAS")
        ENDIF()
        IF("${CBLAS_INCLUDE_PATH}" STREQUAL "CBLAS_INCLUDE_PATH-NOTFOUND")
            set(BLAS_FOUND OFF CACHE BOOL "BLAS_FOUND" FORCE)
        ENDIF()
        IF(BLAS_FOUND)
            MESSAGE(STATUS "BLAS lib:      ${BLAS_LIBRARIES}")
            MESSAGE(STATUS "CBLAS include: ${CBLAS_INCLUDE_PATH}")
            MESSAGE(STATUS "CBLAS define:  ${BLAS_DEFINE}")
        ENDIF()
    ENDIF()

    IF(BLAS_FOUND)
        INCLUDE_DIRECTORIES("${CBLAS_INCLUDE_PATH}")
        ADD_DEFINITIONS("-D${BLAS_DEFINE}")
    ENDIF()

    ADD_EXECUTABLE(
        dash-mxv
        "dash-mxv.cpp")
    TARGET_LINK_LIBRARIES(
        dash-mxv
        "${DASH_LIBRARIES}")
    IF(BLAS_FOUND)
        TARGET_LINK_LIBRARIES(
            dash-mxv
            "${BLAS_LIBRARIES}")
    ENDIF()

    ALPAKA_ADD_EXECUTABLE(
        dash-alpaka-mxv-cpu
        "dash-alpaka-mxv-cpu.cpp")
    TARGET_LINK_LIBRARIES(
        dash-alpaka-mxv-cpu
        PUBLIC "${DASH_LIBRARIES};alpaka")

    IF(ALPAKA_ACC_GPU_CUDA_ENABLE)
        ALPAKA_ADD_EXECUTABLE(
            dash-alpaka-mxv-gpu
            "dash-alpaka-mxv-gpu.cpp")
        TARGET_LINK_LIBRARIES(
            dash-alpaka-mxv-gpu
            PUBLIC "${DASH_LIBRARIES};alpaka")
    ENDIF()
ENDIF()
