CXX = mpinvcc

CUDA_ARCH = compute_35
CUDA_CODE = sm_35

CXXFLAGS += -arch=$(CUDA_ARCH) -code=$(CUDA_CODE) --expt-extended-lambda -lineinfo -Xcompiler "-mno-avx"

FMATH_FLAG = --use_fast_math

LD = mpinvcc
LDFLAGS += -arch=$(CUDA_ARCH) -code=$(CUDA_CODE) -Xcompiler -rdynamic -Xcompiler -flto

HDF5_INC = -I/opt/hdf5_mpich/include/
HDF5_LIB = -L/opt/hdf5_mpich/lib/

INCLUDES +=
LIBS += -lhdf5 -lbfd
