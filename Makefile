# Compiler settings
NVCC = nvcc
NVCC_FLAGS = -Xcompiler -O3 -Xcompiler -Wall -Xptxas -O3 -std=c++17 
LIBS = -lcurand -lcublas

# Default target
all: t t2 debug

# Executables
t: test_async.cu matmul.cu matmul.cuh utils.cuh
	$(NVCC) test_async.cu matmul.cu $(NVCC_FLAGS) $(LIBS) -o ./t

t2: test_managed.cu matmul.cu matmul.cuh utils.cuh
	$(NVCC) test_managed.cu matmul.cu $(NVCC_FLAGS) $(LIBS) -o ./t2

debug: debug_p2p.cu matmul.cu matmul.cuh
	$(NVCC) debug_p2p.cu matmul.cu $(NVCC_FLAGS) $(LIBS) -o ./debug



# Clean rule
clean:
	rm -f t t2 debug

.PHONY: clean 
