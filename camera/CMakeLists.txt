cmake_minimum_required(VERSION 3.10)

set(LIB ocams_camera)

# Source files
file(GLOB SOURCES "*.cpp")

find_package(OpenCV REQUIRED)

# Generate executable and link
include_directories(${CMAKE_CURRENT_SOURCE_DIR} ${OpenCV_INCLUDE_DIRS})
add_library (${LIB} STATIC ${SOURCES})
target_include_directories(${LIB} PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>")
target_link_libraries(${LIB} ${OpenCV_LIBS} pthread udev v4l2)
