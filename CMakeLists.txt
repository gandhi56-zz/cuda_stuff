cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(cuda_stuff LANGUAGES CXX CUDA)

set(SRC_DIR "${PROJECT_SOURCE_DIR}/src")
set(LIB_DIR "${PROJECT_SOURCE_DIR}/lib")
set(BIN_DIR "${PROJECT_SOURCE_DIR}/bin")

file(MAKE_DIRECTORY ${LIB_DIR})
file(MAKE_DIRECTORY ${BIN_DIR})

# VecAdd library ----------------------------------------------------------------- #
add_library(VecAdd SHARED ${SRC_DIR}/VecAdd/VecAdd.cu)
set_target_properties(VecAdd PROPERTIES 
    CUDA_SEPARABLE_COMPILATION ON
    LIBRARY_OUTPUT_DIRECTORY "${LIB_DIR}"
)

# SimpleDeviceQuery library ------------------------------------------------------ #
add_library(SimpleDeviceQuery SHARED ${SRC_DIR}/DeviceQuery/SimpleDeviceQuery.cu)
set_target_properties(SimpleDeviceQuery PROPERTIES 
    CUDA_SEPARABLE_COMPILATION ON
    LIBRARY_OUTPUT_DIRECTORY "${LIB_DIR}"
)

# ImageBlur library ------------------------------------------------------ #
add_library(ImageBlur SHARED ${SRC_DIR}/ImageProcessing/ImageBlur.cu)
set_target_properties(ImageBlur PROPERTIES 
    CUDA_SEPARABLE_COMPILATION ON
    LIBRARY_OUTPUT_DIRECTORY "${LIB_DIR}"
)

# Driver executable -------------------------------------------------------------- #
add_executable(driver ${SRC_DIR}/driver.cpp)
link_directories(driver "${PROJECT_SOURCE_DIR}/lib")
target_link_libraries(driver VecAdd SimpleDeviceQuery ImageBlur )
set_target_properties(driver PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${BIN_DIR}"
)