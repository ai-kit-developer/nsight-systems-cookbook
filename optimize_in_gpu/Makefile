# Makefile for optimize_in_gpu
# Builds all CUDA optimization examples

# Compiler and flags
NVCC = nvcc
NVCC_FLAGS = -O3 -arch=sm_70 -std=c++14

# Directories
SUBDIRS = reduce sgemv sgemm spmv spmm elementwise
BIN_DIRS = $(addsuffix /bin,$(SUBDIRS))

# Default target: build all
.PHONY: all clean $(SUBDIRS)

all: $(SUBDIRS)
	@echo "All projects built successfully!"

# Build individual subdirectories
reduce:
	@echo "Building reduce..."
	@mkdir -p reduce/bin
	@cd reduce && $(MAKE) all

sgemv:
	@echo "Building sgemv..."
	@mkdir -p sgemv/bin
	@$(NVCC) $(NVCC_FLAGS) -o sgemv/bin/Sgemv_v0 sgemv/Sgemv_v0.cu -lcublas
	@$(NVCC) $(NVCC_FLAGS) -o sgemv/bin/Sgemv_v1 sgemv/Sgemv_v1.cu -lcublas
	@$(NVCC) $(NVCC_FLAGS) -o sgemv/bin/Sgemv_v2 sgemv/Sgemv_v2.cu -lcublas
	@$(NVCC) $(NVCC_FLAGS) -o sgemv/bin/ComplexHalfGemv sgemv/ComplexHalfGemv.cu -lcublas

sgemm:
	@echo "Building sgemm..."
	@mkdir -p sgemm/bin
	@$(NVCC) $(NVCC_FLAGS) -o sgemm/bin/sgemm_v1 sgemm/sgemm_v1.cu -lcublas
	@$(NVCC) $(NVCC_FLAGS) -o sgemm/bin/sgemm_v3 sgemm/sgemm_v3.cu -lcublas

spmv:
	@echo "Building spmv..."
	@mkdir -p spmv/bin
	@$(NVCC) $(NVCC_FLAGS) -o spmv/bin/spmv spmv/spmv.cu -lcusparse

spmm:
	@echo "Building spmm..."
	@mkdir -p spmm/bin
	@$(NVCC) $(NVCC_FLAGS) -o spmm/bin/spmm spmm/spmm.cu -lcusparse

elementwise:
	@echo "Building elementwise..."
	@mkdir -p elementwise/bin
	@$(NVCC) $(NVCC_FLAGS) -o elementwise/bin/elementwise_add elementwise/elementwise_add.cu

# Clean all build artifacts
clean:
	@echo "Cleaning all build artifacts..."
	@rm -rf $(BIN_DIRS)
	@echo "Clean completed!"

# Clean individual subdirectories
clean-reduce:
	@rm -rf reduce/bin

clean-sgemv:
	@rm -rf sgemv/bin

clean-sgemm:
	@rm -rf sgemm/bin

clean-spmv:
	@rm -rf spmv/bin

clean-spmm:
	@rm -rf spmm/bin

clean-elementwise:
	@rm -rf elementwise/bin

# Help target
help:
	@echo "Available targets:"
	@echo "  all              - Build all projects (default)"
	@echo "  reduce           - Build reduce project"
	@echo "  sgemv            - Build sgemv project"
	@echo "  sgemm            - Build sgemm project"
	@echo "  spmv             - Build spmv project"
	@echo "  spmm             - Build spmm project"
	@echo "  elementwise      - Build elementwise project"
	@echo "  clean            - Clean all build artifacts"
	@echo "  clean-<project>  - Clean specific project"
	@echo "  help             - Show this help message"

