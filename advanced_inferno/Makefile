CXX ?= icpx
CXXFLAGS ?= -fsycl
INCFLAGS ?= -I../include

SRCS := $(wildcard *.cpp)
PRGS := $(patsubst %.cpp,%,$(SRCS))
RUN :=  $(addprefix run_, $(PRGS))

.PHONY: all
all: $(PRGS)

.PHONY: run_all
run_all: $(sort $(RUN))

%: %.cpp
	$(CXX) $(CXXFLAGS) $(INCFLAGS) -o $@ $^

2_gemm_onemath: 2_gemm_onemath.cpp
	$(CXX) $(CXXFLAGS) $(INCFLAGS) -qmkl -o $@ $^

8_mdspan: 8_mdspan.cpp 
	$(CXX) $(CXXFLAGS) $(INCFLAGS) -I../include/mdspan -o $@ $^

run_%:%
	./$^

.PHONY: clean
clean:
	rm -f $(PRGS)
