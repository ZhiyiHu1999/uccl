# Mirrors experimental/mscclpp/apps/nccl/CMakeLists.txt:
#   mscclpp_nccl_obj  <- GLOB src/*.cu  -> here, all *.cu in this directory
#   mscclpp_nccl      <- SHARED mscclpp_obj + mscclpp_nccl_obj
#   mscclpp_nccl_static
#   mscclpp_audit_nccl <- audit-shim/*.cc

NVCC      ?= nvcc
CXX       ?= g++
AR        ?= ar
MPI_HOME  ?= /usr/mpi/gcc/openmpi-4.1.7rc1
MPI_CXX   ?= $(MPI_HOME)/bin/mpicxx

MINT_DIR  := ..
BUILD_DIR := $(MINT_DIR)/build
CUDA_HOME ?= /usr/local/cuda
CUDA_ARCH ?= 80 90 100

CXX_STD   := -std=c++17
CXXFLAGS  := $(CXX_STD) -Wall -Wextra -fPIC -O2
CUFLAGS   := $(CXX_STD) -Xcompiler -Wall,-Wextra,-fPIC -O2
NVCC_GENCODE := $(foreach arch,$(CUDA_ARCH),-gencode arch=compute_$(arch),code=sm_$(arch))

INCLUDES  := -I. -I$(MINT_DIR)/core -I$(MINT_DIR)/collective -I$(MINT_DIR)/thirdparty \
             -I$(CUDA_HOME)/include -I$(CUDA_HOME)/include/cccl
DEFINES   := -DUSE_IBVERBS -DMSCCLPP_USE_CUDA

OUT_DIR   := build
# allgather_intranode.cu is included by nccl.cu (same anonymous namespace).
# allgather_multinode.cu is compiled as a separate TU.
# ReduceScatter/single-node.cu is included by ReduceScatter/multi-node.cu.
REDUCESCATTER_CU_SRCS := $(filter-out ReduceScatter/single-node.cu,$(wildcard ReduceScatter/*.cu))
LITE_COLL_DIR := $(MINT_DIR)/collective/lite
GPU_DRIVEN_DIR := $(LITE_COLL_DIR)/gpu_driven
CU_SRCS   := $(wildcard *.cu) $(LITE_COLL_DIR)/allgather_multinode.cu $(REDUCESCATTER_CU_SRCS)
CU_OBJS   := $(patsubst %.cu,$(OUT_DIR)/%.cu.o,$(CU_SRCS))

LIB_MINT      := $(BUILD_DIR)/libmint.so
LIB_MINT_COLL := $(BUILD_DIR)/libmint_collective.so

LIB_NCCL_SO  := $(OUT_DIR)/libmscclpp_nccl.so
LIB_NCCL_A   := $(OUT_DIR)/libmscclpp_nccl.a
LIB_AUDIT_SO := $(OUT_DIR)/libmscclpp_audit_nccl.so
AUDIT_OBJ    := $(OUT_DIR)/audit_nccl.o

LDFLAGS := -L$(BUILD_DIR) -L$(CUDA_HOME)/lib64
LDLIBS  := -Wl,--no-as-needed -lmint_collective -lmint -lcudart -lcuda -libverbs -lnuma \
           -lpthread -ldl -lstdc++fs
NVCC_LDLIBS := -Xlinker --no-as-needed -lmint_collective -lmint -lcudart -lcuda \
               -libverbs -lnuma -lpthread -ldl -lstdc++fs

.PHONY: all clean mint-deps device-allgather-compile-test device-collectives-bench

all: mint-deps $(LIB_NCCL_SO) $(LIB_NCCL_A) $(LIB_AUDIT_SO)

mint-deps:
	$(MAKE) -C $(MINT_DIR) core collective

$(LIB_MINT) $(LIB_MINT_COLL): mint-deps
	@:

$(OUT_DIR):
	mkdir -p $@

$(OUT_DIR)/%.cu.o: %.cu | $(OUT_DIR)
	mkdir -p $(@D)
	$(NVCC) $(CUFLAGS) $(NVCC_GENCODE) $(INCLUDES) $(DEFINES) -c $< -o $@

$(OUT_DIR)/allgather.cu.o: allgather.hpp

$(OUT_DIR)/nccl.cu.o: $(LITE_COLL_DIR)/allgather_intranode.cu $(LITE_COLL_DIR)/host_staging_buffer.hpp allgather.hpp lite_common.h

# CpuSwitch is header-based; rebuild its consumers after primitive/channel edits.
CPU_SWITCH_HEADERS := $(wildcard $(LITE_COLL_DIR)/cpu_switch/*.hpp)
$(CU_OBJS): $(CPU_SWITCH_HEADERS)
GPU_DRIVEN_HEADERS := $(wildcard $(GPU_DRIVEN_DIR)/*.cuh) $(wildcard $(GPU_DRIVEN_DIR)/*.inc)
$(OUT_DIR)/nccl.cu.o: $(LITE_COLL_DIR)/cpu_staging_channel.hpp $(LITE_COLL_DIR)/gpu_staging_channel.hpp $(LITE_COLL_DIR)/node_exchange_buffer.hpp $(GPU_DRIVEN_HEADERS)
$(OUT_DIR)/$(LITE_COLL_DIR)/allgather_multinode.cu.o: lite_common.h

$(OUT_DIR)/$(LITE_COLL_DIR)/allgather_multinode.cu.o: $(LITE_COLL_DIR)/allgather_multinode.cu $(LITE_COLL_DIR)/node_exchange_buffer.hpp | $(OUT_DIR)
	mkdir -p $(@D)
	$(NVCC) $(CUFLAGS) $(NVCC_GENCODE) $(INCLUDES) -I. $(DEFINES) -c $< -o $@

$(OUT_DIR)/ReduceScatter/multi-node.cu.o: ReduceScatter/single-node.cu

$(LIB_NCCL_SO): $(CU_OBJS) $(LIB_MINT_COLL) $(LIB_MINT)
	$(CXX) -shared -o $@ $(CU_OBJS) $(LDFLAGS) -Wl,-rpath,'$$ORIGIN:$(abspath $(BUILD_DIR))' $(LDLIBS)

$(LIB_NCCL_A): $(CU_OBJS)
	$(AR) rcs $@ $^

$(AUDIT_OBJ): audit_nccl.cc | $(OUT_DIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(DEFINES) -fPIC -c $< -o $@

$(LIB_AUDIT_SO): $(AUDIT_OBJ)
	$(CXX) -shared -o $@ $^

device-allgather-compile-test: $(OUT_DIR)/gpu_driven/device_collectives_compile_test.cu.o

$(OUT_DIR)/gpu_driven/%.cu.o: $(GPU_DRIVEN_DIR)/%.cu $(GPU_DRIVEN_DIR)/gpu_collectives.cuh | $(OUT_DIR)
	mkdir -p $(@D)
	$(NVCC) $(CUFLAGS) $(NVCC_GENCODE) $(INCLUDES) $(DEFINES) -c $< -o $@

$(OUT_DIR)/device_collectives_bench.o: $(GPU_DRIVEN_DIR)/device_collectives_bench.cu $(GPU_DRIVEN_DIR)/gpu_collectives.cuh | $(OUT_DIR)
	$(NVCC) $(CUFLAGS) -lineinfo $(NVCC_GENCODE) -ccbin $(MPI_CXX) $(INCLUDES) $(DEFINES) -c $< -o $@

$(OUT_DIR)/device_collectives_bench: $(OUT_DIR)/device_collectives_bench.o $(LIB_NCCL_SO)
	$(NVCC) -ccbin $(MPI_CXX) -o $@ $< -L$(OUT_DIR) $(LDFLAGS) \
		-Xlinker -rpath -Xlinker '$$ORIGIN' -lmscclpp_nccl $(NVCC_LDLIBS)

device-collectives-bench: $(OUT_DIR)/device_collectives_bench

clean:
	rm -rf $(OUT_DIR)
