CC     = gcc
NVCC   = nvcc
CFLAGS = -Wall -pedantic -fopenmp
LDFLAGS = -fopenmp -lmpi -lpng -lstdc++
INCLUDES=-I/usr/lib/x86_64-linux-gnu/openmpi/include/
RM     = rm -rf

gpu: dmbrot.cu dmbrot.h
	$(NVCC) -o dmbrot -Xcompiler $(LDFLAGS) $(INCLUDES) dmbrot.cu

.PHONY: test
test: dmbrot_test.cu calc.c
	$(NVCC) -o dmbrot_test -Xcompiler $(LDFLAGS) $(INCLUDES) dmbrot_test.cu
	$(CC) calc.c -o calc -lm

.PHONY: clean
clean:
	$(RM) *.o dmbrot dmbrot_test calc in*.dat resultado.md #cpu gpu
