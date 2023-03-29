cmake_minimum_required(VERSION 3.17)
project(HTCTrackingSys)

if(WIN32)
    include(FetchContent)
    set(FETCHCONTENT_BASE_DIR deps)

    FetchContent_Declare(
        glm
        GIT_REPOSITORY https://github.com/g-truc/glm.git
        GIT_TAG 0.9.9.8
    )
    FetchContent_Declare(
        pybind11
        GIT_REPOSITORY https://github.com/pybind/pybind11.git
        GIT_TAG v2.10.3
    )
    FetchContent_MakeAvailable(glm pybind11)

    find_library(
        SRANIPAL_LIBRARIES
        NAMES SRAnipal
        PATHS "${FETCHCONTENT_BASE_DIR}/SRanipal/lib"
        NO_DEFAULT_PATH NO_CMAKE_FIND_ROOT_PATH
    )
    include_directories(
    "${FETCHCONTENT_BASE_DIR}/SRanipal/include",
    "${CMAKE_BINARY_DIR}/${FETCHCONTENT_BASE_DIR}/glm-src"
    )

    add_library(HTCTrackingSys MODULE htc/HTCTrackingSys.cpp)
    target_link_libraries(HTCTrackingSys
        PRIVATE pybind11::module ${CMAKE_DL_LIBS} ${SRANIPAL_LIBRARIES}
    )
    set_target_properties(HTCTrackingSys PROPERTIES PREFIX "${PYTHON_MODULE_PREFIX}" SUFFIX "${PYTHON_MODULE_EXTENSION}")
endif()
