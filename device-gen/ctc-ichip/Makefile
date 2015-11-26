CXX = g++-4.9
CXXFLAGS +=  -O0 -g3 -std=c++11 -fopenmp

ctc-ichip:	*.cpp collage.o redistance.o 2Dto3D.o
	$(CXX) $(CXXFLAGS) -I../../mpi-dpd/ collage.o redistance.o 2Dto3D.o main.cpp -o ctc-ichip

collage.o: ../common/collage.h ../common/collage.cpp
	$(CXX) $(CXXFLAGS) -c $^

redistance.o: ../common/redistance.h ../common/redistance.cpp
	$(CXX) $(CXXFLAGS) -c $^

2Dto3D.o: ../common/2Dto3D.h ../common/2Dto3D.cpp
	$(CXX) $(CXXFLAGS) -c $^

clean:
	rm -f ctc-ichip *.o *.d *.h.gch


