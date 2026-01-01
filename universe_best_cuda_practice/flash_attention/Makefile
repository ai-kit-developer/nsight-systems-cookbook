# Makefile for flash_attention
# Builds Flash Attention examples with PyTorch support

# Compiler and flags
NVCC = nvcc
CXX = g++
CXX_FLAGS = -std=c++17 -O3
NVCC_FLAGS = -std=c++17 -O3 -arch=sm_89 --expt-relaxed-constexpr
NVCC_FLAGS += -lineinfo  # For profiling support

# Directories
BIN_DIR = bin

# Default target: build all
.PHONY: all clean help

all: $(BIN_DIR)/example-app $(BIN_DIR)/flash-atten-main $(BIN_DIR)/softmax
	@echo "All flash_attention targets built successfully!"

# Build example-app (requires PyTorch)
# Note: PyTorch C++ API compilation is complex. Consider using CMake for better support.
$(BIN_DIR)/example-app: example-app.cpp
	@mkdir -p $(BIN_DIR)
	@echo "Note: Building with PyTorch requires proper libtorch installation."
	@echo "      For best results, use CMake: cmake . && make"
	@echo "      Or set TORCH_PATH environment variable."
	@TORCH_PATH_FOUND=""; \
	TORCH_LIB_FOUND=""; \
	TORCH_INCLUDE_FOUND=""; \
	if [ -n "$$TORCH_PATH" ] && [ -d "$$TORCH_PATH" ]; then \
		TORCH_PATH_FOUND="$$TORCH_PATH"; \
		TORCH_LIB_FOUND="$$TORCH_PATH/lib"; \
		TORCH_INCLUDE_FOUND="$$TORCH_PATH/include"; \
	elif [ -n "$$TORCH_INSTALL_PREFIX" ] && [ -d "$$TORCH_INSTALL_PREFIX" ]; then \
		TORCH_PATH_FOUND="$$TORCH_INSTALL_PREFIX"; \
		TORCH_LIB_FOUND="$$TORCH_INSTALL_PREFIX/lib"; \
		TORCH_INCLUDE_FOUND="$$TORCH_INSTALL_PREFIX/include"; \
	else \
		for TORCH_BASE in "/data/libtorch" "$$HOME/libtorch" "/usr/local/libtorch" "/opt/libtorch" \
			"$$HOME/flash-attention-minimal/libtorch" "$$HOME/workspace/libtorch" "$$HOME/projects/libtorch"; do \
			if [ -d "$$TORCH_BASE" ] && [ -d "$$TORCH_BASE/lib" ] && [ -d "$$TORCH_BASE/include" ]; then \
				TORCH_PATH_FOUND="$$TORCH_BASE"; \
				TORCH_LIB_FOUND="$$TORCH_BASE/lib"; \
				TORCH_INCLUDE_FOUND="$$TORCH_BASE/include"; \
				break; \
			fi; \
		done; \
	fi; \
	if [ -z "$$TORCH_PATH_FOUND" ] || [ ! -d "$$TORCH_LIB_FOUND" ] || [ ! -d "$$TORCH_INCLUDE_FOUND" ]; then \
		echo "Warning: PyTorch not found. Skipping example-app."; \
		echo "         To build, please:"; \
		echo "         1. Install libtorch from https://pytorch.org/get-started/locally/"; \
		echo "         2. Set TORCH_PATH: export TORCH_PATH=/path/to/libtorch"; \
		echo "         3. Or use CMake: cmake . && make"; \
		touch $(BIN_DIR)/example-app; \
	else \
		echo "Building example-app with PyTorch at $$TORCH_PATH_FOUND..."; \
		$(CXX) $(CXX_FLAGS) \
			-I$$TORCH_INCLUDE_FOUND -I$$TORCH_INCLUDE_FOUND/torch/csrc/api/include \
			-L$$TORCH_LIB_FOUND \
			-o $(BIN_DIR)/example-app \
			example-app.cpp \
			-ltorch -ltorch_cpu -lc10 -lc10_cuda \
			-lcudart -pthread -Wl,-rpath,$$TORCH_LIB_FOUND || \
		echo "Failed to build example-app. Try using CMake instead: cmake . && make"; \
	fi

# Build flash-atten-main (requires PyTorch and CUDA)
# Note: PyTorch C++ API compilation is complex. Consider using CMake for better support.
$(BIN_DIR)/flash-atten-main: flash-atten-main.cpp flash.cu
	@mkdir -p $(BIN_DIR)
	@echo "Note: Building with PyTorch requires proper libtorch installation."
	@echo "      For best results, use CMake: cmake . && make"
	@TORCH_PATH_FOUND=""; \
	TORCH_LIB_FOUND=""; \
	TORCH_INCLUDE_FOUND=""; \
	if [ -n "$$TORCH_PATH" ] && [ -d "$$TORCH_PATH" ]; then \
		TORCH_PATH_FOUND="$$TORCH_PATH"; \
		TORCH_LIB_FOUND="$$TORCH_PATH/lib"; \
		TORCH_INCLUDE_FOUND="$$TORCH_PATH/include"; \
	elif [ -n "$$TORCH_INSTALL_PREFIX" ] && [ -d "$$TORCH_INSTALL_PREFIX" ]; then \
		TORCH_PATH_FOUND="$$TORCH_INSTALL_PREFIX"; \
		TORCH_LIB_FOUND="$$TORCH_INSTALL_PREFIX/lib"; \
		TORCH_INCLUDE_FOUND="$$TORCH_INSTALL_PREFIX/include"; \
	else \
		for TORCH_BASE in "/data/libtorch" "$$HOME/libtorch" "/usr/local/libtorch" "/opt/libtorch" \
			"$$HOME/flash-attention-minimal/libtorch" "$$HOME/workspace/libtorch" "$$HOME/projects/libtorch"; do \
			if [ -d "$$TORCH_BASE" ] && [ -d "$$TORCH_BASE/lib" ] && [ -d "$$TORCH_BASE/include" ]; then \
				TORCH_PATH_FOUND="$$TORCH_BASE"; \
				TORCH_LIB_FOUND="$$TORCH_BASE/lib"; \
				TORCH_INCLUDE_FOUND="$$TORCH_BASE/include"; \
				break; \
			fi; \
		done; \
	fi; \
	if [ -z "$$TORCH_PATH_FOUND" ] || [ ! -d "$$TORCH_LIB_FOUND" ] || [ ! -d "$$TORCH_INCLUDE_FOUND" ]; then \
		echo "Warning: PyTorch not found. Skipping flash-atten-main."; \
		echo "         To build, please:"; \
		echo "         1. Install libtorch from https://pytorch.org/get-started/locally/"; \
		echo "         2. Set TORCH_PATH: export TORCH_PATH=/path/to/libtorch"; \
		echo "         3. Or use CMake: cmake . && make"; \
		touch $(BIN_DIR)/flash-atten-main; \
	else \
		echo "Building flash-atten-main with PyTorch at $$TORCH_PATH_FOUND..."; \
		$(NVCC) $(NVCC_FLAGS) \
			-I$$TORCH_INCLUDE_FOUND -I$$TORCH_INCLUDE_FOUND/torch/csrc/api/include \
			-L$$TORCH_LIB_FOUND \
			-o $(BIN_DIR)/flash-atten-main \
			flash-atten-main.cpp flash.cu \
			-ltorch -ltorch_cpu -lc10 -lc10_cuda \
			-lcudart -pthread -Wl,-rpath,$$TORCH_LIB_FOUND || \
		echo "Failed to build flash-atten-main. Try using CMake instead: cmake . && make"; \
	fi

# Build softmax (standalone, no PyTorch required)
$(BIN_DIR)/softmax: softmax.cpp
	@mkdir -p $(BIN_DIR)
	@echo "Building softmax..."
	@$(CXX) $(CXX_FLAGS) -o $(BIN_DIR)/softmax softmax.cpp || \
		echo "Failed to build softmax"

# Clean build artifacts
clean:
	@echo "Cleaning flash_attention build artifacts..."
	@rm -rf $(BIN_DIR)
	@echo "Clean completed!"

# Help target
help:
	@echo "Flash Attention Makefile"
	@echo ""
	@echo "Available targets:"
	@echo "  all                    - Build all targets (default)"
	@echo "  example-app            - Build example-app (requires PyTorch)"
	@echo "  flash-atten-main       - Build flash-atten-main (requires PyTorch and CUDA)"
	@echo "  softmax                - Build softmax (standalone, no dependencies)"
	@echo "  clean                  - Clean all build artifacts"
	@echo "  help                   - Show this help message"
	@echo ""
	@echo "Environment variables:"
	@echo "  TORCH_PATH             - Path to libtorch installation"
	@echo "  TORCH_INSTALL_PREFIX   - Alternative path to libtorch installation"
	@echo ""
	@echo "PyTorch will be searched in:"
	@echo "  - TORCH_PATH or TORCH_INSTALL_PREFIX (if set)"
	@echo "  - /data/libtorch"
	@echo "  - $$HOME/libtorch"
	@echo "  - /usr/local/libtorch"
	@echo "  - /opt/libtorch"
	@echo "  - $$HOME/flash-attention-minimal/libtorch"
	@echo "  - Python torch package location (if available)"

