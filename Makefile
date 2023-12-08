# Compiler settings
NVCC = nvcc
NVCC_FLAGS = -Xcompiler -O3 -Xcompiler -Wall -Xptxas -O3 -std=c++17 
LIBS = -lcurand -lcublas #-lcnpy -lz

all: t t2 test_time

# Executables
t: test_async.cu matmul.cu matmul.cuh utils.cuh
	$(NVCC) test_async.cu matmul.cu $(NVCC_FLAGS) $(LIBS) -o ./t

t2: test_managed.cu matmul.cu matmul.cuh utils.cuh
	$(NVCC) test_managed.cu matmul.cu $(NVCC_FLAGS) $(LIBS) -o ./t2


test_time: test_time.cu matmul.cu matmul.cuh utils.cuh
	$(NVCC) test_time.cu matmul.cu $(NVCC_FLAGS) $(LIBS) -o ./test_time

# Clean rule
clean:
	rm -f t t2 

.PHONY: clean 