cmake_minimum_required(VERSION 2.6)
project(ProBLM)
set(CMAKE_CXX_STANDARD 17)

set (V3D_DIR ${CMAKE_SOURCE_DIR}/Libs/V3D/)
include (${V3D_DIR}/Config/v3d_macros.cmake)
include (${V3D_DIR}/Config/local_config.cmake)

set (EXTRA_INC_DIRS /usr/local/include
        /usr/include/
        /usr/local/include/eigen3    
        ${V3D_DIR}/
        ./include/  
    )

set (SOURCES 
        src/twoview_models.cpp
        src/nlsq.cpp
        src/stochastic_nlsq.cpp
        src/robust_nlsq.cpp
        src/robust_nlsq_gnc.cpp
    )

set (CMAKE_BUILD_TYPE Release)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
    set(CMAKE_CXX_FLAGS "-O3")
endif()

include_directories( ${V3D_INCLUDE_DIRS} ${EXTRA_INC_DIRS})
link_directories(${V3D_DIR}/)

add_executable(problm main.cpp ${SOURCES})
target_link_libraries(problm) 

add_executable(homography homography_registration.cpp ${SOURCES})
target_link_libraries(homography libV3D.a png z jpeg) 
