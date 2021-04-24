cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(cuda_stuff LANGUAGES CXX CUDA)

set(SRC_DIR "${PROJECT_SOURCE_DIR}/src")
set(LIB_DIR "${PROJECT_SOURCE_DIR}/lib")
set(BIN_DIR "${PROJECT_SOURCE_DIR}/bin")

add_library(VecAdd SHARED ${SRC_DIR}/VecAdd/VecAdd.cu)
set_target_properties(VecAdd PROPERTIES 
    CUDA_SEPARABLE_COMPILATION ON
    LIBRARY_OUTPUT_DIRECTORY "${LIB_DIR}"
)

add_executable(driver ${SRC_DIR}/driver.cpp)
link_directories(driver "${PROJECT_SOURCE_DIR}/lib")
target_link_libraries(driver VecAdd)
set_target_properties(driver PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${BIN_DIR}"
)
