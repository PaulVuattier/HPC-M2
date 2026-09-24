cmake_minimum_required(VERSION 3.16)
project(boris_pusher LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_executable(boris_pusher src/main.cpp)

add_executable(pic_pusher src/pic_main.cpp)

# Field-solver unit tests on analytic sine fields (used by notebooks/pic_tests).
add_executable(pic_unit src/pic_unit.cpp)

target_include_directories(boris_pusher PRIVATE include)
target_include_directories(pic_pusher PRIVATE include)
target_include_directories(pic_unit PRIVATE include)

if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
  # -ffp-contract=off: keep a*b+c from fusing into an FMA. The array/single
  # equivalence test demands bitwise equality, and a vectorised push that
  # contracts differently from the scalar one would differ in the last ulp.
  target_compile_options(boris_pusher PRIVATE -Wall -Wextra -O2 -ffp-contract=off)
  target_compile_options(pic_pusher PRIVATE -Wall -Wextra -O2 -ffp-contract=off)
  target_compile_options(pic_unit PRIVATE -Wall -Wextra -O2 -ffp-contract=off)
endif()

enable_testing()
# Array independence: boris_push_array must equal N separate boris_push calls.
add_test(NAME array_selftest COMMAND boris_pusher --selftest-array)
# PIC guards: conservation, uniform gather, Faraday/divB, uniform equivalence.
add_test(NAME pic_selftest_1d COMMAND pic_pusher --selftest-pic dim=1)
add_test(NAME pic_selftest_2d COMMAND pic_pusher --selftest-pic dim=2)
