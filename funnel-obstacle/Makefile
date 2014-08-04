CXXFLAGS +=  -Ofast -DNDEBUG

test:	*.cpp *.h funnel-obstacle.o
	$(CXX) $(CXXFLAGS) funnel-obstacle.o obst-check.cpp -o test

funnel-obstacle.o: funnel-obstacle.cpp funnel-obstacle.h
	$(CXX) $(CXXFLAGS) -c $^
