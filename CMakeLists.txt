cmake_minimum_required(VERSION 2.6)

project(STOCHAS_LM)
set(CMAKE_CXX_STANDARD 17)

find_package(OpenCV)

set(CMAKE_EXPORT_COMPILE_COMMANDS "ON")

set (V3D_DIR /home/intellhave/Dropbox/RobBundle/SSBA-4.0/)
set (V3D_INCLUDE_DIRS ${V3D_DIR}/.)

include (${V3D_DIR}/Config/v3d_macros.cmake)
include (${V3D_DIR}/Config/local_config.cmake)


set (EXTRA_INC_DIRS /usr/local/include
    /usr/include/x86_64-linux-gnu/c++/7.5.0
    /usr/include/c++/7.5.0/
    /usr/include/c++/7.5.0/bits/
    /usr/include/
    /usr/local/include/eigen3    
    ./include/  
    ./libs/LBFGS/include/
    )

set (EXTRA_LINK_DIRS /usr/local/lib/)


# set (SOURCES test_essential.cpp)
set (SOURCES src/image_utils.cpp
    src/eth3d_handler.cpp
    src/twoview_models.cpp
    src/nlsq.cpp
    src/stochastic_nlsq.cpp
    src/stochastic_nlsq_momentum.cpp
    src/robust_nlsq.cpp
    src/robust_stochastic_nlsq.cpp
    src/robust_nlsq_gnc.cpp
    src/robust_stochastic_nlsq_gnc.cpp
    src/robust_nlsq_irls.cpp
    src/lbfgs_nlsq.cpp
    )



set (CMAKE_BUILD_TYPE Release)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
    set(CMAKE_CXX_FLAGS "-O3")
endif()

# set(CMAKE_CXX_FLAGS "-Wall -Wextra")
# set(CMAKE_CXX_FLAGS_DEBUG "-g")
# set(CMAKE_CXX_FLAGS_RELEASE "-O3")


include_directories( ${V3D_INCLUDE_DIRS} ${EXTRA_INC_DIRS})
link_directories(${V3D_DIR}/build ${EXTRA_LINK_DIRS})


add_executable(main main.cpp ${SOURCES})
target_link_libraries(main ${OpenCV_LIBS})

add_executable(robust_homography robust_homography_registration.cpp ${SOURCES})
target_link_libraries(robust_homography ${OpenCV_LIBS} V3D png z jpeg colamd)  

