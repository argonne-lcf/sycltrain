CXX ?= dpcpp
CCFLAGS ?= -std=c++17 

SRCS := $(wildcard *.cpp)
PRGS := $(patsubst %.cpp,%,$(SRCS))

all: $(PRGS) 
%: %.cpp
	-$(CXX) $(CCFLAGS) -o $@ $^

clean:
	rm -f $(PRGS)
