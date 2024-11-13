# host compiler
HOST_COMPILER ?= g++
CUDA_PATH ?= /usr/local/cuda
NVCC          := /usr/local/cuda/bin/nvcc -ccbin $(HOST_COMPILER)


# internal flags
NVCCFLAGS   := -m$(shell getconf LONG_BIT)
CCFLAGS     :=
LDFLAGS     :=

ALL_CCFLAGS :=
ALL_CCFLAGS += $(NVCCFLAGS)
ALL_CCFLAGS += $(addprefix -Xcompiler ,$(CCFLAGS))

ALL_LDFLAGS :=
ALL_LDFLAGS += $(ALL_CCFLAGS)
ALL_LDFLAGS += $(addprefix -Xlinker ,$(LDFLAGS))

# Common includes and paths for CUDA
INCLUDES  := -I$(CUDA_PATH)/include/ 
LIBRARIES := -lcublas 
LIBRARIES += -L${CUDA_PATH}/lib64

INCLUDES  += -I/home/ferrar/cutlass/include -I/home/ferrar/cutlass/tools/util/include

################################################################################

# Gencode arguments
SMS ?= 90a
# Generate SASS code for each SM architecture listed in $(SMS)
$(foreach sm,$(SMS),$(eval GENCODE_FLAGS += -gencode arch=compute_$(sm),code=sm_$(sm)))

ALL_CCFLAGS += --threads 0 --std=c++11 -O3 -lineinfo
# ALL_CCFLAGS += -maxrregcount=255
ALL_CCFLAGS += --use_fast_math
ALL_CCFLAGS += --ptxas-options=-v,-warn-lmem-usage,--warn-on-spills
################################################################################
# Target rules
all: hopper_warp_specialized_gemm


hopper_warp_specialized_gemm.o: 48_hopper_warp_specialized_gemm.cu helper.h
	$(EXEC) $(NVCC) $(INCLUDES) $(ALL_CCFLAGS) $(GENCODE_FLAGS) -o $@ -c $<

hopper_warp_specialized_gemm: hopper_warp_specialized_gemm.o
	$(EXEC) $(NVCC) $(ALL_LDFLAGS) $(GENCODE_FLAGS) -o $@ $+ $(LIBRARIES) 

clean:
	rm -f hopper_warp_specialized_gemm hopper_warp_specialized_gemm.o