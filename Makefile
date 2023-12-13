# Compiler settings
NVCC = nvcc
NVCC_FLAGS = -Xcompiler -O3 -Xcompiler -Wall -Xptxas -O3 -std=c++17 
LIBS = -lcurand -lcublas #-lcnpy -lz

all: async managed manual

# Executables
./async: test_async.cu matmul.cu matmul.cuh utils.cuh
	$(NVCC) test_async.cu matmul.cu $(NVCC_FLAGS) $(LIBS) -o ./async
./managed: test_time.cu matmul.cu matmul.cuh utils.cuh
	$(NVCC) test_time.cu matmul.cu $(NVCC_FLAGS) $(LIBS) -o ./managed
./manual: test_manual.cu matmul.cu matmul.cuh utils.cuh
	$(NVCC) test_manual.cu matmul.cu $(NVCC_FLAGS) $(LIBS) -o ./manual

# test_time: test_time.cu matmul.cu matmul.cuh utils.cuh
#	 $(NVCC) test_time.cu matmul.cu $(NVCC_FLAGS) $(LIBS) -o ./test_time

# test_mlp: test_mlp.cu matmul.cu matmul.cuh utils.cuh network.cuh
#	 $(NVCC) test_mlp.cu matmul.cu $(NVCC_FLAGS) $(LIBS) -o ./test_mlp
# Clean rule
clean:
	rm -f t t2 

.PHONY: clean 