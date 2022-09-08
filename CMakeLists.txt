cmake_minimum_required(VERSION 3.14)
# project(Bino_VO)

set(CMAKE_BUILD_TYPE "Debug")
# set(CMAKE_BUILD_TYPE "Release")
# set(CMAKE_CXX_FLAGS "-std=c++11")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17")
set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -O0")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O0")

list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

find_package(Pangolin REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)

find_package(Sophus REQUIRED )
find_package(G2O REQUIRED)

include_directories(${Sophus_INCLUDE_DIRS})
include_directories(${G2O_INCLUDE_DIR})

add_executable(VO_Sparse_DirTrack VO_Sparse_Direct_Track.cpp)
add_executable(VO_Sparse_Multi_DirTrack VO_Multi_Sparse_Dir_Track.cpp)

target_link_libraries(VO_Sparse_DirTrack PUBLIC ${OpenCV_LIBS}
                       ${Pangolin_LIBRARIES} ${Sophus_LIBRARIES} ${G2O_CORE_LIBRARY} ${G2O_STUFF_LIBRARY} g2o_types_slam3d fmt glog)
target_link_libraries(VO_Sparse_DirTrack PUBLIC /home/libo/oneTBB-2019_U8/build/my_tbb_build_release/libtbb.so)

target_link_libraries(VO_Sparse_Multi_DirTrack PUBLIC ${OpenCV_LIBS}
                       ${Pangolin_LIBRARIES} ${Sophus_LIBRARIES} ${G2O_CORE_LIBRARY} ${G2O_STUFF_LIBRARY} g2o_types_slam3d fmt glog)
target_link_libraries(VO_Sparse_Multi_DirTrack PUBLIC /home/libo/oneTBB-2019_U8/build/my_tbb_build_release/libtbb.so)