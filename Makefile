# Compiler settings
NVCC = nvcc
NVCC_FLAGS = -Xcompiler -O3 -Xcompiler -Wall -Xptxas -O3 -std=c++17 
LIBS = -lcurand -lcublas -lcnpy -lz -lnccl

all: async managed manual test_mlp

# Executables
./async: test_async.cu include/matmul.cu include/matmul.cuh include/utils.cuh
	$(NVCC) test_async.cu include/matmul.cu $(NVCC_FLAGS) $(LIBS) -o ./async
./managed: test_time.cu include/matmul.cu include/matmul.cuh include/utils.cuh
	$(NVCC) test_time.cu include/matmul.cu $(NVCC_FLAGS) $(LIBS) -o ./managed
./manual: test_manual.cu include/matmul.cu include/matmul.cuh include/utils.cuh
	$(NVCC) test_manual.cu include/matmul.cu $(NVCC_FLAGS) $(LIBS) -o ./manual

test_mlp: test_mlp.cu include/matmul.cu include/matmul.cuh include/utils.cuh include/network.cuh
	 $(NVCC) test_mlp.cu include/matmul.cu $(NVCC_FLAGS) $(LIBS) -o ./test_mlp
# Clean rule
clean:
	rm -f ./async ./managed ./manual ./test_mlp

.PHONY: clean 