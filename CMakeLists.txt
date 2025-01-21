cmake_minimum_required(VERSION 3.10)
project(GPUObjectDetection)

set(CMAKE_CXX_STANDARD 14)

find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(gpu_object_detection src/main.cpp)

target_link_libraries(gpu_object_detection ${OpenCV_LIBS})
