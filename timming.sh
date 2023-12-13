# nvcc test_time.cu matmul.cu -Xcompiler -O3 -Xcompiler -Wall -Xptxas -O3 -std c++17 -lcurand -lcublas -o managed
# nvcc test_manual.cu matmul.cu -Xcompiler -O3 -Xcompiler -Wall -Xptxas -O3 -std c++17 -lcurand -lcublas -o manual
# nvcc test_async.cu matmul.cu -Xcompiler -O3 -Xcompiler -Wall -Xptxas -O3 -std c++17 -lcurand -lcublas -o async

# Just make so you don't recompile every time
make 
for ((power = 5; power <= 11; power++)); do
    argument=$((2**power))
    argumentwo=$((3**power))
    ./managed $argument $argument $argumentwo 2
    ./manual $argument $argument $argumentwo 2
    # ./async $argument 2
done
