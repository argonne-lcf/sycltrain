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

%_gemm_onemath: %_gemm_onemath.cpp
	$(CXX) $(CXXFLAGS) $(INCFLAGS) -qmkl -o $@ $^

%_mdspan: %_mdspan.cpp
	$(CXX) $(CXXFLAGS) $(INCFLAGS) -I../include/mdspan -o $@ $^

run_%:%
	./$^

.PHONY: clean
clean:
	rm -f $(PRGS)
