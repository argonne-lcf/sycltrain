CXX ?= dpcpp 
CXXLAGS ?= -std=c++17

SRCS := $(wildcard *.cpp)
PRGS := $(patsubst %.cpp,%,$(SRCS))

all: $(PRGS) 
%: %.cpp
	-$(CXX) -o $@ $^ $(CXXFLAGS)

9_matrix_mul_mkl: 9_matrix_mul_mkl.cpp
	-$(CXX) -o $@  $^ $(CXXFLAGS) -DMKL_ILP64 -lmkl_sycl -lmkl_intel_ilp64 -lmkl_sequential -lmkl_core

clean:
	rm -f $(PRGS)
