# Compiler settings
NVCC = nvcc
NVCC_FLAGS = -Xcompiler -O3 -Xcompiler -Wall -Xptxas -O3 -std=c++17 
DEBUG_FLAGS = -Xcompiler -Wall -std=c++17 -g -G
LIBS = -lcurand -lcublas -lcnpy -lz -lnccl

all: async managed manual test_mlp

# Executables
./async: test_async.cu src/matmul.cu src/matmul.cuh src/utils.cuh
	$(NVCC) test_async.cu src/matmul.cu $(NVCC_FLAGS) $(LIBS) -o ./async
./managed: test_time.cu src/matmul.cu src/matmul.cuh src/utils.cuh
	$(NVCC) test_time.cu src/matmul.cu $(NVCC_FLAGS) $(LIBS) -o ./managed
./manual: test_manual.cu src/matmul.cu src/matmul.cuh src/utils.cuh
	$(NVCC) test_manual.cu src/matmul.cu $(NVCC_FLAGS) $(LIBS) -o ./manual

./test_mlp: test_mlp.cu src/matmul.cu src/matmul.cuh src/utils.cuh src/network.cuh
	 $(NVCC) test_mlp.cu src/matmul.cu $(NVCC_FLAGS) $(LIBS) -o ./test_mlp
./debug: test_mlp.cu src/matmul.cu src/matmul.cuh src/utils.cuh src/network.cuh
	 $(NVCC) test_mlp.cu src/matmul.cu $(DEBUG_FLAGS) $(LIBS) -o ./test_mlp

# Clean rule
clean:
	rm -f ./async ./managed ./manual ./test_mlp ./test_time ./test

.PHONY: clean 