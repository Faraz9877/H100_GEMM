compile:
	nvcc -arch=sm_90a -lcuda -std=c++17 matmul_h100.cu -o test

run:
	./test