cmake_minimum_required(VERSION 3.10)

project(VitisStereoCPUVer LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs calib3d)

add_executable(vitis_bm_cpu
    src/vitis_bm_cpu.cpp
)

add_executable(vitis_sgbm_cpu
    src/vitis_sgbm_cpu.cpp
)

foreach(target vitis_bm_cpu vitis_sgbm_cpu)
    target_include_directories(${target} PRIVATE ${OpenCV_INCLUDE_DIRS})
    target_link_libraries(${target} PRIVATE ${OpenCV_LIBS})
    target_compile_options(${target} PRIVATE -Wall -Wextra -Wpedantic)
endforeach()

message(STATUS "OpenCV version: ${OpenCV_VERSION}")
message(STATUS "OpenCV include directories: ${OpenCV_INCLUDE_DIRS}")
