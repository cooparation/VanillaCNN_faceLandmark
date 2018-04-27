cmake_minimum_required(VERSION 2.8.12)
project(examples)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -fPIC")

# caffe
set(Caffe_DIR /home/liusj/dl/caffe/.build_release)
find_package(Caffe)
include_directories(${Caffe_DIR}/include)

# dlib
set(DLIB_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../dlib)
include_directories(${DLIB_DIR})

# opencv
find_package(OpenCV REQUIRED)
include_directories(include
    ${OpenCV_INCLUDE_DIRS}
    )

set(SRC_FILES ./face_landmark.cpp
    )
add_executable(face_landmark ${SRC_FILES})
target_link_libraries(face_landmark
    ${OpenCV_LIBS}
    ${Caffe_LIBRARIES}
    ${DLIB_DIR}/examples/build/dlib_build/libdlib.a
    )

set(SRC_FILES ./face_cnn.cpp
    )
add_executable(face_cnn ${SRC_FILES})
target_link_libraries(face_cnn
    ${OpenCV_LIBS}
    ${Caffe_LIBRARIES}
    ${DLIB_DIR}/examples/build/dlib_build/libdlib.a
    )
