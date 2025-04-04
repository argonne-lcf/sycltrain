CXX ?= icpx
CXXFLAGS ?= -std=c++17 -fsycl
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

run_%:%
	./$^

.PHONY: clean
clean:
	rm -f $(PRGS)
