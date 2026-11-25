# Author: Nguyen Huu Khanh Hung - 23120271 (Member 3). Portable V0 build; CUDA path remains available.
NVCC ?= nvcc
CXX = g++
TARGET ?= stitch
SRCS = main.cu stb_impl.cpp v0_frontend.cpp v0_backend.cpp v0_stitching.cpp
HEADERS = common.h versions.h pipeline_config.h hd_math.h hd_backend.h frontend_math.h brief_pattern.h logging.h stb_image.h stb_image_write.h
CUDA_ARCH ?= sm_75
CUDA_FLAGS = -O3 -std=c++17 --fmad=false -arch=$(CUDA_ARCH)
CUDA_SRCS = gpu_frontend.cu gpu_backend.cu gpu_pipeline.cu
CUDA_HEADERS = gpu_frontend.cuh gpu_pipeline.cuh gpu_backend.cuh device_buffer.cuh cuda_utils.cuh
BACKEND_TEST ?= test_backend
BACKEND_SRCS = tests/test_backend.cu gpu_backend.cu stb_impl.cpp v0_frontend.cpp v0_backend.cpp v0_stitching.cpp
BACKEND_HEADERS = gpu_backend.cuh device_buffer.cuh cuda_utils.cuh
all: cpu
cpu: $(SRCS) $(HEADERS)
	$(CXX) -O3 -std=c++14 -x c++ $(SRCS) -o $(TARGET)
cuda: $(SRCS) $(CUDA_SRCS) $(HEADERS) $(CUDA_HEADERS)
	$(NVCC) $(CUDA_FLAGS) -DSTITCH_CUDA $(SRCS) $(CUDA_SRCS) -o $(TARGET)
test-backend: $(BACKEND_SRCS) $(BACKEND_HEADERS) $(HEADERS)
	$(NVCC) $(CUDA_FLAGS) $(BACKEND_SRCS) -o $(BACKEND_TEST)
clean:
	rm -f $(TARGET) $(TARGET).exe $(BACKEND_TEST) *.o *.obj
.PHONY: all cpu cuda test-backend clean
