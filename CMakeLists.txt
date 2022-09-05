cmake_minimum_required(VERSION 3.14)
# project(Bino_VO)

set(CMAKE_BUILD_TYPE "Debug")
# set(CMAKE_BUILD_TYPE "Release")
# set(CMAKE_CXX_FLAGS "-std=c++11")
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
# include_directories("/usr/include/eigen3")

add_executable(Bino_VO dense_depth.cpp)
add_executable(GN_test GN_BA.cpp)
add_executable(inter_frame inter_frame_match.cpp)
add_executable(VO_Track VO_Track.cpp)
add_executable(Traj_Plot traj_plot.cpp)
target_link_libraries(Bino_VO ${OpenCV_LIBS} ${Pangolin_LIBRARIES})
target_link_libraries(GN_test ${OpenCV_LIBS} ${Pangolin_LIBRARIES} ${Sophus_LIBRARIES} fmt)

target_link_libraries(inter_frame PUBLIC ${OpenCV_LIBS}
                      ${Pangolin_LIBRARIES} ${Sophus_LIBRARIES} ${G2O_CORE_LIBRARY} ${G2O_STUFF_LIBRARY} g2o_types_slam3d fmt glog)

target_link_libraries(VO_Track PUBLIC ${OpenCV_LIBS}
                      ${Pangolin_LIBRARIES} ${Sophus_LIBRARIES} ${G2O_CORE_LIBRARY} ${G2O_STUFF_LIBRARY} g2o_types_slam3d fmt glog)

target_link_libraries(Traj_Plot PUBLIC ${OpenCV_LIBS}
                      ${Pangolin_LIBRARIES} ${Sophus_LIBRARIES} ${G2O_CORE_LIBRARY} ${G2O_STUFF_LIBRARY} g2o_types_slam3d fmt glog)