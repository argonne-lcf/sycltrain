CXX ?= icpx
CXXFLAGS ?= -fsycl
INCFLAGS ?= -I../include -I../include/mdspan/

SRCS := $(wildcard *.cpp)
PRGS := $(patsubst %.cpp,%,$(SRCS))
RUN :=  $(addprefix run_, $(PRGS))

.PHONY: run_all
run_all: $(sort $(RUN))

.PHONY: all
all: $(PRGS)

%: %.cpp
	$(CXX) $(CXXFLAGS) $(INCFLAGS) -o $@ $^

9_matrix_mul_mkl: 9_matrix_mul_mkl.cpp
	-$(CXX) -o $@  $^ $(CXXFLAGS) -DMKL_ILP64 -qmkl

run_%:%
	./$^

.PHONY: clean
clean:
	rm -f $(PRGS)
