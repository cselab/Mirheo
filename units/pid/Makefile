CXX      = g++
CXXFLAGS = -O3 -Wall -Wpedantic
INCFLAGS = -I ../../src/plugins
LDFLAGS  = 

EXE = step

all: $(EXE)

%: %.cpp
	$(CXX) $(CXXFLAGS) $(INCFLAGS) $< -o $@ $(LDFLAGS)

clean:; rm -rf $(EXE)

.PHONY: clean
