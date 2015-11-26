CXX ?= CC

ply2vtk: main.cpp
	$(CXX) -Ofast -std=c++11 -fopenmp main.cpp \
	-I/apps/daint/VTK/6.2/gnu_491/include/vtk-6.2 -L/apps/daint/VTK/6.2/gnu_491/lib \
	-lvtkIOImage-6.2 -lvtkCommonDataModel-6.2 -lvtkpng-6.2  -lvtktiff-6.2 \
	-lvtkmetaio-6.2 -lvtkDICOMParser-6.2 -lvtkzlib-6.2 -lvtksys-6.2 \
	-lvtkIOXMLParser-6.2 -lvtkCommonExecutionModel-6.2 -lvtkCommonTransforms-6.2 \
	-lvtkCommonCore-6.2 -lvtkIOXML-6.2 -lvtkexpat-6.2  -lvtkjpeg-6.2 -lvtkIOCore-6.2 \
	-lvtkCommonSystem-6.2 -lvtkCommonTransforms-6.2 -lvtkCommonMath-6.2 \
	-lvtkCommonMisc-6.2 \
	-o ply2vtk

clean: 
	rm -f ply2vtk

.PHONY = clean
