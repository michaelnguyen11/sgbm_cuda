cmake_minimum_required(VERSION 3.10)

PROJECT(sgbm_cuda)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)

add_subdirectory(camera)

find_package(OpenCV REQUIRED)
find_package(CUDA REQUIRED)

include_directories(${CMAKE_CURRENT_SOURCE_DIR} ${OpenCV_INCLUDE_DIRS} ${CUDA_INCLUDE_DIRS})

link_directories(${OpenCV_LIBRARY_DIRS})
link_directories(${CUDA_LIBRARY_DIRS})

FILE(GLOB SRC_FILES *.cpp)
FILE(GLOB CUDA_FILES *.cu)
FILE(GLOB HDR_FILES *.h*)

set(
	CUDA_NVCC_FLAGS
	${CUDA_NVCC_FLAGS};
	-O3
	#-gencode arch=compute_20,code=[sm_20,sm_21] 666 CHECK THIS DEPEND ON THE PLATFORM
#	-gencode arch=compute_30,code=sm_30
#	-gencode arch=compute_35,code=sm_35
#	-gencode arch=compute_50,code=[sm_50,compute_50]
#	-gencode arch=compute_52,code=[sm_52,compute_52]
	-gencode arch=compute_61,code=sm_61
#	-gencode arch=compute_62,code=sm_62
#	-G
)

cuda_add_executable(sgbm_cuda ${CUDA_FILES} ${HDR_FILES} ${SRC_FILES})

TARGET_LINK_LIBRARIES(sgbm_cuda ocams_camera ${OpenCV_LIBS} ${CUDA_CUDA_LIBRARY} ${CUDA_CUDART_LIBRARY})

file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/camera/ocams_calibration_720p.xml DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
