override CXXFLAGS += -std=c++11 -g -Ofast -fopenmp

test: main.cpp
	$(CXX) $(CXXFLAGS) main.cpp -o test

clean:
	rm -f test

.PHONY = clean
