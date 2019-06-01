CC     = gcc
NVCC   = nvcc
CFLAGS = -Wall -pedantic -fopenmp
LDFLAGS = -fopenmp -lpng -lstdc++
RM     = rm -rf
print  = echo

gpu: mbrot.cu mbrot.h
	$(NVCC) -o mbrot -Xcompiler $(LDFLAGS) mbrot.cu

.PHONY: test
test: mbrot_test.cu calc.c
	$(NVCC) -o mbrot_test -Xcompiler $(LDFLAGS) mbrot_test.cu
	$(CC) calc.c -o calc -lm
.PHONY: clean
clean:
	$(RM) *.o mbrot mbrot_test calc in*.dat resultado.md #cpu gpu
