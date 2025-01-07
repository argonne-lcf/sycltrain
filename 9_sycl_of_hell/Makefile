CXX ?= icpx
CXXFLAGS ?= -fsycl
INCFLAGS ?= -I./3rd_party

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
