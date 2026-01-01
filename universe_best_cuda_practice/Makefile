# Makefile for universe_best_cuda_practice
# Builds all CUDA optimization examples

# Compiler and flags
NVCC = nvcc
NVCC_FLAGS = -O3 -arch=sm_89 -std=c++17
NVCC_FLAGS += -lineinfo  # For profiling support
LDFLAGS = -lcublas -lcudart

# Directories
SUBDIRS = 1_cuda_reduce_study 2_cuda_sgemm_study 3_kernel_profiling_guide \
          4_tensor_core_wmma 5_mma_and_swizzle 6_cutlass_study flash_attention

# Default target: build all
.PHONY: all clean $(SUBDIRS) help

all: $(SUBDIRS)
	@echo "All projects built successfully!"

# Build individual subdirectories
1_cuda_reduce_study:
	@echo "Building 1_cuda_reduce_study..."
	@mkdir -p 1_cuda_reduce_study/bin
	@$(NVCC) $(NVCC_FLAGS) -o 1_cuda_reduce_study/bin/my_reduce_v0_global_memory \
		1_cuda_reduce_study/my_reduce_v0_global_memory.cu $(LDFLAGS)
	@$(NVCC) $(NVCC_FLAGS) -o 1_cuda_reduce_study/bin/my_reduce_v1_shared_memory \
		1_cuda_reduce_study/my_reduce_v1_shared_memory.cu $(LDFLAGS)
	@$(NVCC) $(NVCC_FLAGS) -o 1_cuda_reduce_study/bin/my_reduce_v2_no_divergence_branch \
		1_cuda_reduce_study/my_reduce_v2_no_divergence_branch.cu $(LDFLAGS)
	@$(NVCC) $(NVCC_FLAGS) -o 1_cuda_reduce_study/bin/my_reduce_v3_no_bank_conflict \
		1_cuda_reduce_study/my_reduce_v3_no_bank_conflict.cu $(LDFLAGS)
	@$(NVCC) $(NVCC_FLAGS) -o 1_cuda_reduce_study/bin/my_reduce_v4_add_during_load_plan_a \
		1_cuda_reduce_study/my_reduce_v4_add_during_load_plan_a.cu $(LDFLAGS)
	@$(NVCC) $(NVCC_FLAGS) -o 1_cuda_reduce_study/bin/my_reduce_v4_add_during_load_plan_b \
		1_cuda_reduce_study/my_reduce_v4_add_during_load_plan_b.cu $(LDFLAGS)
	@$(NVCC) $(NVCC_FLAGS) -o 1_cuda_reduce_study/bin/my_reduce_v5_unroll_last_warp \
		1_cuda_reduce_study/my_reduce_v5_unroll_last_warp.cu $(LDFLAGS)
	@$(NVCC) $(NVCC_FLAGS) -o 1_cuda_reduce_study/bin/my_reduce_v6_completely_unroll \
		1_cuda_reduce_study/my_reduce_v6_completely_unroll.cu $(LDFLAGS)
	@$(NVCC) $(NVCC_FLAGS) -o 1_cuda_reduce_study/bin/my_reduce_v7_mutli_add \
		1_cuda_reduce_study/my_reduce_v7_mutli_add.cu $(LDFLAGS)
	@$(NVCC) $(NVCC_FLAGS) -o 1_cuda_reduce_study/bin/my_reduce_v8_shuffle \
		1_cuda_reduce_study/my_reduce_v8_shuffle.cu $(LDFLAGS)

2_cuda_sgemm_study:
	@echo "Building 2_cuda_sgemm_study..."
	@mkdir -p 2_cuda_sgemm_study/bin
	@$(NVCC) $(NVCC_FLAGS) -o 2_cuda_sgemm_study/bin/my_sgemm_v0_global_memory \
		2_cuda_sgemm_study/my_sgemm_v0_global_memory.cu $(LDFLAGS)
	@$(NVCC) $(NVCC_FLAGS) -o 2_cuda_sgemm_study/bin/my_sgemm_v1_shared_memory \
		2_cuda_sgemm_study/my_sgemm_v1_shared_memory.cu $(LDFLAGS)
	@$(NVCC) $(NVCC_FLAGS) -o 2_cuda_sgemm_study/bin/my_sgemm_v2_shared_memory_sliding_windows \
		2_cuda_sgemm_study/my_sgemm_v2_shared_memory_sliding_windows.cu $(LDFLAGS)
	@$(NVCC) $(NVCC_FLAGS) -o 2_cuda_sgemm_study/bin/my_sgemm_v3_increase_work_of_per_thread \
		2_cuda_sgemm_study/my_sgemm_v3_increase_work_of_per_thread.cu $(LDFLAGS)
	@$(NVCC) $(NVCC_FLAGS) -o 2_cuda_sgemm_study/bin/my_sgemm_v4_using_float4 \
		2_cuda_sgemm_study/my_sgemm_v4_using_float4.cu $(LDFLAGS)
	@$(NVCC) $(NVCC_FLAGS) -o 2_cuda_sgemm_study/bin/my_sgemm_v5_register_outer_product \
		2_cuda_sgemm_study/my_sgemm_v5_register_outer_product.cu $(LDFLAGS)
	@$(NVCC) $(NVCC_FLAGS) -o 2_cuda_sgemm_study/bin/my_sgemm_v6_register_outer_product_float4 \
		2_cuda_sgemm_study/my_sgemm_v6_register_outer_product_float4.cu $(LDFLAGS)
	@$(NVCC) $(NVCC_FLAGS) -o 2_cuda_sgemm_study/bin/my_sgemm_v7_A_smem_transpose \
		2_cuda_sgemm_study/my_sgemm_v7_A_smem_transpose.cu $(LDFLAGS)
	@$(NVCC) $(NVCC_FLAGS) -o 2_cuda_sgemm_study/bin/my_sgemm_v8_double_buffer \
		2_cuda_sgemm_study/my_sgemm_v8_double_buffer.cu $(LDFLAGS)

3_kernel_profiling_guide:
	@echo "Building 3_kernel_profiling_guide..."
	@mkdir -p 3_kernel_profiling_guide/bin
	@$(NVCC) $(NVCC_FLAGS) -o 3_kernel_profiling_guide/bin/transpose \
		3_kernel_profiling_guide/transpose.cu $(LDFLAGS)
	@$(NVCC) $(NVCC_FLAGS) -o 3_kernel_profiling_guide/bin/combined_access \
		3_kernel_profiling_guide/combined_access.cu $(LDFLAGS)
	@$(NVCC) $(NVCC_FLAGS) -o 3_kernel_profiling_guide/bin/my_transpose_v1_naive \
		3_kernel_profiling_guide/my_transpose_v1_naive.cu $(LDFLAGS)
	@$(NVCC) $(NVCC_FLAGS) -o 3_kernel_profiling_guide/bin/my_transpose_v2_float4 \
		3_kernel_profiling_guide/my_transpose_v2_float4.cu $(LDFLAGS)
	@$(NVCC) $(NVCC_FLAGS) -o 3_kernel_profiling_guide/bin/my_transpose_v3_float2 \
		3_kernel_profiling_guide/my_transpose_v3_float2.cu $(LDFLAGS)
	@$(NVCC) $(NVCC_FLAGS) -o 3_kernel_profiling_guide/bin/my_transpose_v4_float2_1x2 \
		3_kernel_profiling_guide/my_transpose_v4_float2_1x2.cu $(LDFLAGS)
	@$(NVCC) $(NVCC_FLAGS) -o 3_kernel_profiling_guide/bin/my_transpose \
		3_kernel_profiling_guide/my_transpose.cu $(LDFLAGS)
	@$(NVCC) $(NVCC_FLAGS) -o 3_kernel_profiling_guide/bin/roofline_model \
		3_kernel_profiling_guide/roofline_model.cu $(LDFLAGS)
	@$(NVCC) $(NVCC_FLAGS) -o 3_kernel_profiling_guide/bin/my_transpose_v5_shared_memory \
		3_kernel_profiling_guide/my_transpose_v5_shared_memory.cu $(LDFLAGS)
	@$(NVCC) $(NVCC_FLAGS) -o 3_kernel_profiling_guide/bin/my_transpose_v6_no_bank_conflict \
		3_kernel_profiling_guide/my_transpose_v6_no_bank_conflict.cu $(LDFLAGS)
	@$(NVCC) $(NVCC_FLAGS) -o 3_kernel_profiling_guide/bin/my_transpose_v7_increase_work_of_per_thread \
		3_kernel_profiling_guide/my_transpose_v7_increase_work_of_per_thread.cu $(LDFLAGS)

4_tensor_core_wmma:
	@echo "Building 4_tensor_core_wmma..."
	@mkdir -p 4_tensor_core_wmma/bin
	@$(NVCC) $(NVCC_FLAGS) -o 4_tensor_core_wmma/bin/hgemm_v1_wmma_m16n16k16_naive_kernel \
		4_tensor_core_wmma/hgemm_v1_wmma_m16n16k16_naive_kernel.cu $(LDFLAGS)
	@$(NVCC) $(NVCC_FLAGS) -o 4_tensor_core_wmma/bin/hgemm_v2_wmma_m16n16k16_mma4x2_kernel \
		4_tensor_core_wmma/hgemm_v2_wmma_m16n16k16_mma4x2_kernel.cu $(LDFLAGS)
	@$(NVCC) $(NVCC_FLAGS) -o 4_tensor_core_wmma/bin/hgemm_v3_wmma_m16n16k16_mma4x2_warp2x4_kernel \
		4_tensor_core_wmma/hgemm_v3_wmma_m16n16k16_mma4x2_warp2x4_kernel.cu $(LDFLAGS)
	@$(NVCC) $(NVCC_FLAGS) -o 4_tensor_core_wmma/bin/hgemm_v4_wmma_m16n16k16_mma4x2_warp2x4_dbuf_async_kernel \
		4_tensor_core_wmma/hgemm_v4_wmma_m16n16k16_mma4x2_warp2x4_dbuf_async_kernel.cu $(LDFLAGS)

5_mma_and_swizzle:
	@echo "Building 5_mma_and_swizzle..."
	@mkdir -p 5_mma_and_swizzle/bin
	@$(NVCC) $(NVCC_FLAGS) -o 5_mma_and_swizzle/bin/v1_simple_wmma \
		5_mma_and_swizzle/v1_simple_wmma.cu $(LDFLAGS)
	@$(NVCC) $(NVCC_FLAGS) -o 5_mma_and_swizzle/bin/v2_shared_memory_wmma \
		5_mma_and_swizzle/v2_shared_memory_wmma.cu $(LDFLAGS)
	@$(NVCC) $(NVCC_FLAGS) -o 5_mma_and_swizzle/bin/hgemm_v1_mma_m16n8k16_naive_kernel \
		5_mma_and_swizzle/hgemm_v1_mma_m16n8k16_naive_kernel.cu $(LDFLAGS)
	@$(NVCC) $(NVCC_FLAGS) -o 5_mma_and_swizzle/bin/v3_shared_memory_wmma_padding \
		5_mma_and_swizzle/v3_shared_memory_wmma_padding.cu $(LDFLAGS)
	@$(NVCC) $(NVCC_FLAGS) -o 5_mma_and_swizzle/bin/v4_shared_memory_mma \
		5_mma_and_swizzle/v4_shared_memory_mma.cu $(LDFLAGS)
	@$(NVCC) $(NVCC_FLAGS) -o 5_mma_and_swizzle/bin/v5_shared_memory_mma_swizzle \
		5_mma_and_swizzle/v5_shared_memory_mma_swizzle.cu $(LDFLAGS)

6_cutlass_study:
	@echo "Building 6_cutlass_study..."
	@echo "Note: This requires CUTLASS library. Please set CUTLASS_PATH if needed."
	@mkdir -p 6_cutlass_study/bin
	@CUTLASS_PATH_USE=$${CUTLASS_PATH:-/home/hongkailin/cutlass/include/}; \
	CUTLASS_UTIL_PATH_USE=$${CUTLASS_UTIL_PATH:-/home/hongkailin/cutlass/tools/util/include}; \
	echo "Using CUTLASS_PATH=$$CUTLASS_PATH_USE"; \
	echo "Using CUTLASS_UTIL_PATH=$$CUTLASS_UTIL_PATH_USE"; \
	$(NVCC) $(NVCC_FLAGS) --expt-relaxed-constexpr \
		-I$$CUTLASS_PATH_USE -I$$CUTLASS_UTIL_PATH_USE \
		-o 6_cutlass_study/bin/v1_print_half \
		6_cutlass_study/v1_print_half.cu $(LDFLAGS) || \
		echo "Failed to build v1_print_half (CUTLASS may not be available)"; \
	$(NVCC) $(NVCC_FLAGS) --expt-relaxed-constexpr \
		-I$$CUTLASS_PATH_USE -I$$CUTLASS_UTIL_PATH_USE \
		-o 6_cutlass_study/bin/v2_gemm_kernel \
		6_cutlass_study/v2_gemm_kernel.cu $(LDFLAGS) || \
		echo "Failed to build v2_gemm_kernel (CUTLASS may not be available)"; \
	$(NVCC) $(NVCC_FLAGS) --expt-relaxed-constexpr -O0 \
		-I$$CUTLASS_PATH_USE -I$$CUTLASS_UTIL_PATH_USE \
		-o 6_cutlass_study/bin/v3_turing_tensorop_gemm \
		6_cutlass_study/v3_turing_tensorop_gemm.cu $(LDFLAGS) || \
		echo "Failed to build v3_turing_tensorop_gemm (CUTLASS may not be available)"

flash_attention:
	@echo "Building flash_attention..."
	@echo "Note: This requires PyTorch. Using CMake is recommended for this project."
	@mkdir -p flash_attention/bin
	@echo "Flash attention requires PyTorch and should be built with CMake."
	@echo "Please use: cd flash_attention && cmake . && make"

# Clean all build artifacts
clean:
	@echo "Cleaning all build artifacts..."
	@rm -rf 1_cuda_reduce_study/bin 2_cuda_sgemm_study/bin \
		3_kernel_profiling_guide/bin 4_tensor_core_wmma/bin \
		5_mma_and_swizzle/bin 6_cutlass_study/bin flash_attention/bin
	@echo "Clean completed!"

# Clean individual subdirectories
clean-1_cuda_reduce_study:
	@rm -rf 1_cuda_reduce_study/bin

clean-2_cuda_sgemm_study:
	@rm -rf 2_cuda_sgemm_study/bin

clean-3_kernel_profiling_guide:
	@rm -rf 3_kernel_profiling_guide/bin

clean-4_tensor_core_wmma:
	@rm -rf 4_tensor_core_wmma/bin

clean-5_mma_and_swizzle:
	@rm -rf 5_mma_and_swizzle/bin

clean-6_cutlass_study:
	@rm -rf 6_cutlass_study/bin

clean-flash_attention:
	@rm -rf flash_attention/bin

# Help target
help:
	@echo "Available targets:"
	@echo "  all                        - Build all projects (default)"
	@echo "  1_cuda_reduce_study         - Build reduce study project"
	@echo "  2_cuda_sgemm_study          - Build sgemm study project"
	@echo "  3_kernel_profiling_guide    - Build kernel profiling guide project"
	@echo "  4_tensor_core_wmma          - Build tensor core wmma project"
	@echo "  5_mma_and_swizzle           - Build mma and swizzle project"
	@echo "  6_cutlass_study             - Build cutlass study project (requires CUTLASS)"
	@echo "  flash_attention             - Flash attention (requires PyTorch, use CMake)"
	@echo "  clean                       - Clean all build artifacts"
	@echo "  clean-<project>              - Clean specific project"
	@echo "  help                        - Show this help message"
	@echo ""
	@echo "Environment variables:"
	@echo "  CUTLASS_PATH                - Path to CUTLASS include directory"
	@echo "  CUTLASS_UTIL_PATH           - Path to CUTLASS util include directory"

