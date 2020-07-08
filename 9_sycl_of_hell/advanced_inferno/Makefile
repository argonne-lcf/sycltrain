CC ?= dpcpp 
CCFLAGS ?= -std=c++17

SRCS := $(wildcard *.cpp)
PRGS := $(patsubst %.cpp,%,$(SRCS))

all: $(PRGS) 
%: %.cpp
	-$(CC) -o $@ $^ $(CCFLAGS)

10_matrix_mul_mkl: 10_matrix_mul_mkl.cpp
	-$(CC) -o $@  $^ $(CCFLAGS) -DMKL_ILP64 -lmkl_sycl -lmkl_intel_ilp64 -lmkl_sequential -lmkl_core

clean:
	rm -f $(PRGS)
