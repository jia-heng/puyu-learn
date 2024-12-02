cmake_minimum_required(VERSION 3.15)
project(DEMO LANGUAGES CXX)

# 设置 C++ 标准
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 设置输出目录
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR})

set(OPEN_SOURCE_DIR ${PROJECT_SOURCE_DIR}/../open_source)
# 添加包含目录
include_directories(
    ${OPEN_SOURCE_DIR}/TensorRT10.5/include
    ${OPEN_SOURCE_DIR}/TensorRT10.5/samples/utils
    ${OPEN_SOURCE_DIR}/TensorRT10.5/samples
    ${OPEN_SOURCE_DIR}/openexr2.2/include
    ${CUDA_TOOLKIT_ROOT_DIR}/include
)

# 添加库目录
link_directories(
    ${CUDA_TOOLKIT_ROOT_DIR}/lib64
    ${PROJECT_SOURCE_DIR}/../open_source/openexr2.2/lib
)

# 添加源文件
set(SOURCE_FILES
    demo.cpp
	dataloader.cpp
    ${OPEN_SOURCE_DIR}/TensorRT10.5/samples/common/getopt.c
    ${OPEN_SOURCE_DIR}/TensorRT10.5/samples/common/logger.cpp
    ${OPEN_SOURCE_DIR}/TensorRT10.5/samples/utils/fileLock.cpp
    ${OPEN_SOURCE_DIR}/TensorRT10.5/samples/utils/timingCache.cpp
)

# 添加定义
add_definitions(-DOPENEXR_DLL)

# 创建可执行文件
add_executable(${PROJECT_NAME} ${SOURCE_FILES})

# 添加库依赖
target_link_libraries(${PROJECT_NAME}
    nvinfer_10
    nvinfer_plugin_10
    nvonnxparser_10
    cudart
    Half
    Iex-2_2
    IlmImf-2_2
    Imath-2_2
    # 其他依赖库
    kernel32
    user32
    gdi32
    winspool
    comdlg32
    advapi32
    shell32
    ole32
    oleaut32
    uuid
    odbc32
    odbccp32
)

# 设置编译器警告
if(MSVC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /wd4244 /wd4996")
endif()

# 设置 Runtime Library
set_target_properties(${PROJECT_NAME} PROPERTIES
    MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>"
)

# 设置字符集
add_compile_definitions(_MBCS)

# 设置调试信息
if(${CMAKE_BUILD_TYPE} STREQUAL "Debug")
    target_compile_options(${PROJECT_NAME} PRIVATE "/Zi")
    target_link_options(${PROJECT_NAME} PRIVATE "/DEBUG")
endif()

# 设置平台工具集
set(CMAKE_GENERATOR_PLATFORM x64)
set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")