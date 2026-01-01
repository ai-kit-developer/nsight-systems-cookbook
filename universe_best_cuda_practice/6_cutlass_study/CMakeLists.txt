# Try to find CUTLASS from environment variable or common locations
if(DEFINED ENV{CUTLASS_PATH} AND EXISTS "$ENV{CUTLASS_PATH}")
    set(CUTLASS_PATH "$ENV{CUTLASS_PATH}")
    if(DEFINED ENV{CUTLASS_UTIL_PATH} AND EXISTS "$ENV{CUTLASS_UTIL_PATH}")
        set(CUTLASS_UTIL_PATH "$ENV{CUTLASS_UTIL_PATH}")
    else()
        # Try to infer util path from CUTLASS_PATH
        get_filename_component(CUTLASS_BASE_DIR "$ENV{CUTLASS_PATH}" DIRECTORY)
        get_filename_component(CUTLASS_BASE_DIR "${CUTLASS_BASE_DIR}" DIRECTORY)
        if(EXISTS "${CUTLASS_BASE_DIR}/tools/util/include")
            set(CUTLASS_UTIL_PATH "${CUTLASS_BASE_DIR}/tools/util/include")
        endif()
    endif()
else()
    # Try common installation locations
    set(CUTLASS_SEARCH_PATHS
        "/data/cutlass"
        "$ENV{HOME}/cutlass"
        "/usr/local/cutlass"
        "/opt/cutlass"
        "$ENV{HOME}/workspace/cutlass"
        "$ENV{HOME}/projects/cutlass"
    )
    
    foreach(CUTLASS_BASE ${CUTLASS_SEARCH_PATHS})
        if(EXISTS "${CUTLASS_BASE}/include" AND EXISTS "${CUTLASS_BASE}/tools/util/include")
            set(CUTLASS_PATH "${CUTLASS_BASE}/include")
            set(CUTLASS_UTIL_PATH "${CUTLASS_BASE}/tools/util/include")
            break()
        endif()
    endforeach()
endif()

# Warn if CUTLASS not found
if(NOT DEFINED CUTLASS_PATH OR NOT EXISTS "${CUTLASS_PATH}")
    message(WARNING "CUTLASS not found. Please set CUTLASS_PATH environment variable or install CUTLASS in one of the common locations.")
    message(STATUS "Searched in: /data/cutlass, $ENV{HOME}/cutlass, /usr/local/cutlass, /opt/cutlass")
endif()
set(CUTLASS_NVCC_ARCHS 89)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr")

add_executable(v1_print_half v1_print_half.cu)
target_link_libraries(v1_print_half PRIVATE CUDA::cudart ${CUDA_cublas_LIBRARY})
target_include_directories(v1_print_half PRIVATE ${CUTLASS_PATH})
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(v1_print_half PRIVATE  $<$<COMPILE_LANGUAGE:CUDA>:$<$<CONFIG:Debug>:-O0 -G>>)
else()
    target_compile_options(v1_print_half PRIVATE -lineinfo)
endif()

add_executable(v2_gemm_kernel v2_gemm_kernel.cu)
target_link_libraries(v2_gemm_kernel PRIVATE CUDA::cudart ${CUDA_cublas_LIBRARY})
target_include_directories(v2_gemm_kernel PRIVATE ${CUTLASS_PATH} ${CUTLASS_UTIL_PATH})
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(v2_gemm_kernel PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-G>)
else()
    target_compile_options(v2_gemm_kernel PRIVATE -lineinfo)
endif()

add_executable(v3_turing_tensorop_gemm v3_turing_tensorop_gemm.cu)
target_link_libraries(v3_turing_tensorop_gemm PRIVATE CUDA::cudart ${CUDA_cublas_LIBRARY})
target_include_directories(v3_turing_tensorop_gemm PRIVATE ${CUTLASS_PATH} ${CUTLASS_UTIL_PATH})
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(v3_turing_tensorop_gemm PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-g -G>)
else()
    target_compile_options(v3_turing_tensorop_gemm PRIVATE -lineinfo)
endif()
target_compile_options(v3_turing_tensorop_gemm PRIVATE -O0)