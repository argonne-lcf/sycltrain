CC=dpcpp
CCFLAGS= -std=c++17 

SRCS := $(wildcard *.cpp)
PRGS := $(patsubst %.cpp,%,$(SRCS))

all: $(PRGS) 
%: %.cpp
	-$(CC) -o $@ $^ $(CCFLAGS)

clean:
	rm -f $(PRGS)
