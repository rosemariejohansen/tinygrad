cmake_minimum_required(VERSION 3.20)

project(tinygrad
    VERSION 0.1.0
    LANGUAGES CXX
)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_executable(tinygrad
    src/main.cpp
    src/scalar.cpp
    src/tensor.cpp
)

target_include_directories(tinygrad
    PRIVATE
        ${PROJECT_SOURCE_DIR}/include
)

enable_testing()

add_executable(scalar_tests
    tests/scalar_tests.cpp
    src/scalar.cpp
)

target_include_directories(scalar_tests
    PRIVATE
        ${PROJECT_SOURCE_DIR}/include
)

add_test(
    NAME scalar_tests
    COMMAND scalar_tests
)

add_executable(tensor_tests
    tests/tensor_tests.cpp
    src/tensor.cpp
)

target_include_directories(tensor_tests
    PRIVATE
        ${PROJECT_SOURCE_DIR}/include
)

add_test(
    NAME tensor_tests
    COMMAND tensor_tests
)
