# VERSION 7
CUDA=/usr/local/cuda
NVCC=${CUDA}/bin/nvcc
IDIR=${CUDA}/include
LDIR=${CUDA}/lib64
WARN=-Wno-deprecated-gpu-targets
NVCC_FLAGS=-I${IDIR} -L${LDIR} ${WARN}

# make dbg=1 tells nvcc to add debugging symbols to the binary
ifeq ($(dbg),1)
	NVCC_FLAGS += -g -O0
else
	NVCC_FLAGS += -O3
endif

# make emu=1 compiles the CUDA kernels for emulation
ifeq ($(emu),1)
	NVCC_FLAGS += -deviceemu
endif

BINARIES = vector_max breadth_first_search forward_subst 2d_convolve

all: $(BINARIES)

vector_max: vector_max.cu
	$(NVCC) $(NVCC_FLAGS) vector_max.cu -o vector_max -lcuda

breadth_first_search: breadth_first_search.cu
	$(NVCC) $(NVCC_FLAGS) breadth_first_search.cu -o breadth_first_search -lcuda

forward_subst: forward_subst.cu
	$(NVCC) $(NVCC_FLAGS) forward_subst.cu -o forward_subst -lcuda

2d_convolve: 2d_convolve.cu
	$(NVCC) $(NVCC_FLAGS) 2d_convolve.cu -o 2d_convolve -lcuda


clean:
	rm -f *.o *~ $(BINARIES)

.PHONY: clean all
