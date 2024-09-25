
cmake_minimum_required(VERSION 3.10)

# Project name 
project(openacc_tut)

# 檢查編譯器是否支持 OpenACC
if (CMAKE_CXX_COMPILER_ID STREQUAL "PGI")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -acc -gpu=cc86 -Minfo")  # 設置 OpenACC 標誌
else()
    message(FATAL_ERROR "This project requires a compiler that supports OpenACC.")
endif()

# Find OpenCV package
find_package(OpenCV REQUIRED)

add_executable(${PROJECT_NAME} main.cpp)

target_include_directories(${PROJECT_NAME} PRIVATE 
    ${OpenCV_INCLUDE_DIRS})

# Link the OpenCV libraries
target_link_libraries(${PROJECT_NAME} 
    ${OpenCV_LIBS})
