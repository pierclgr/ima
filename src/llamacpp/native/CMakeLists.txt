cmake_minimum_required(VERSION 3.20)
project(ima_llamacpp_worker LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_RPATH "${LLAMA_CPP_DIR}/build/bin")

foreach(component llama mtmd llama-common ggml ggml-base)
    find_library(${component}_LIBRARY NAMES ${component}
        PATHS "${LLAMA_CPP_DIR}/build/bin" NO_DEFAULT_PATH REQUIRED)
    list(APPEND native_libraries "${${component}_LIBRARY}")
endforeach()

add_executable(ima-llamacpp-worker worker.cpp)
target_include_directories(ima-llamacpp-worker PRIVATE
    "${LLAMA_CPP_DIR}/include"
    "${LLAMA_CPP_DIR}/ggml/include"
    "${LLAMA_CPP_DIR}/common")
target_include_directories(ima-llamacpp-worker PRIVATE
    "${LLAMA_CPP_DIR}/tools/mtmd")
target_link_libraries(ima-llamacpp-worker PRIVATE ${native_libraries})
