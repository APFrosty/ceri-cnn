cmake_minimum_required(VERSION 3.18)
project(ceri-cnn)
set(CMAKE_CXX_STANDARD 20)

include_directories(include)
link_directories(/usr/lib)

# Executable and source
file(GLOB SOURCE_FILES ./src/host/*.cpp)
add_executable(ceri-cnn ${SOURCE_FILES})

# Libraries
target_link_libraries(ceri-cnn OpenCL.so)
