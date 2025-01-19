# Minimum CMake version
cmake_minimum_required(VERSION 3.14)

# Project name and version
project(torchplusplus 
    VERSION 1.0.0
    DESCRIPTION "A C++ deep learning framework"
    LANGUAGES CXX
)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Build type configuration
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Include directories
include_directories(${PROJECT_SOURCE_DIR}/include)

# Collect source files
file(GLOB_RECURSE LIB_SOURCES
    "${PROJECT_SOURCE_DIR}/src/*.cpp"
)

# Create the main library
add_library(${PROJECT_NAME} ${LIB_SOURCES})

# Enable testing
enable_testing()

# Download and configure GoogleTest
include(FetchContent)
FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG release-1.12.1
)
FetchContent_MakeAvailable(googletest)

# Add subdirectories
add_subdirectory(tests)
add_subdirectory(examples)

# Output configuration
message(STATUS "")
message(STATUS "Configuration for ${PROJECT_NAME}:")
message(STATUS "  Version:          ${PROJECT_VERSION}")
message(STATUS "  Build type:       ${CMAKE_BUILD_TYPE}")
message(STATUS "  C++ Standard:     ${CMAKE_CXX_STANDARD}")
message(STATUS "  Compiler:         ${CMAKE_CXX_COMPILER_ID}")
message(STATUS "  Install prefix:   ${CMAKE_INSTALL_PREFIX}")
message(STATUS "")