export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$PWD/Library:$LD_LIBRARY_PATH
export TEST_SYSTEM_PARAMS=0
ldd ./xhpl
./xhpl HPL.dat
