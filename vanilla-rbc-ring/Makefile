override CXXFLAGS += -O2 -g -Wall -c -fmessage-length=0 -std=c++11 -MMD 

test: rbc_ring.cpp
	$(CXX) $(CXXFLAGS) rbc_ring.cpp -o test

clean:
	rm -f test

.PHONY = clean
