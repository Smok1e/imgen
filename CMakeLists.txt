cmake_minimum_required(VERSION 3.22 FATAL_ERROR)

project(ImGen LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED True)

set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED True)
set(CMAKE_CUDA_ARCHITECTURES native)

#======================================== External packages

find_package(CUDAToolkit REQUIRED)
find_package(glm CONFIG REQUIRED)
find_package(Stb REQUIRED)
find_package(SFML COMPONENTS system window graphics CONFIG REQUIRED)

add_subdirectory("external/pl_mpeg")

#======================================== Main

add_executable(
	imgen
	"src/main.cpp"
	"src/font.cpp"
	"src/palette.cpp"
	"src/cuda_error.cpp" 
	"src/kernels.cu"
)

add_custom_target(
	assets
    COMMAND 
	${CMAKE_COMMAND} -E copy_directory 
	${CMAKE_CURRENT_LIST_DIR}/assets 
	${CMAKE_CURRENT_BINARY_DIR}/assets
)

target_include_directories(
	imgen 
	PRIVATE 
	"include/"
	${Stb_INCLUDE_DIR}
	${CUDAToolkit_INCLUDE_DIRS}
)

target_link_libraries(
	imgen 
	PRIVATE 
	pl_mpeg
	CUDA::cudart
	sfml-system 
	sfml-graphics 
	sfml-window
)

target_compile_definitions(
	imgen 
	PRIVATE 
	$<$<COMPILE_LANGUAGE:CUDA>:GLM_FORCE_CUDA>
)

add_dependencies(imgen assets)

#========================================