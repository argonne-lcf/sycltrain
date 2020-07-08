CXX = clang++ 
CCFLAGS = -std=c++17 -fsycl -lOpenCL 

SRCS := $(wildcard *.cpp)
PRGS := $(patsubst %.cpp,%,$(SRCS))
RUN :=  $(addprefix run_, $(PRGS))

.PHONY: run_all
run_all: $(sort $(RUN))

.PHONY: all
all: $(PRGS)

%: %.cpp
	$(CXX) $(CCFLAGS) -o $@ $^

run_%:%
	./$^

.PHONY: clean
clean:
	rm -f $(PRGS)
