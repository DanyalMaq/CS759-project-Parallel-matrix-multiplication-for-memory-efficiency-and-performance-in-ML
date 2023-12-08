NVCC = nvcc
NVCC_FLAGS = -Xcompiler -O3 -Xcompiler -Wall -Xptxas -O3 -std=c++17 
LIBS = -lcurand -lcublas #-lcnpy -lz

all: debug p2p_time

debug: debug_p2p.cu ../matmul.cu ../matmul.cuh ../utils.cuh
	$(NVCC) debug_p2p.cu ../matmul.cu $(NVCC_FLAGS) $(LIBS) -o ./debug

p2p_time: p2p_time.cu ../matmul.cu ../matmul.cuh ../utils.cuh
	$(NVCC) p2p_time.cu ../matmul.cu $(NVCC_FLAGS) $(LIBS) -o ./p2p_time

clean :
	rm -f debug p2p_time