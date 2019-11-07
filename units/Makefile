all: build

BUILD=build

LOCAL_CMAKE_FLAGS = "-Dgtest_disable_pthreads=ON"
LOCAL_CMAKE_FLAGS += "-DBUILD_GMOCK=OFF"
LOCAL_CMAKE_FLAGS += "-DINSTALL_GTEST=OFF"

build:
	(mkdir -p $(BUILD) && \
	 cd $(BUILD) && cmake ${CMAKE_FLAGS} $(LOCAL_CMAKE_FLAGS) .. && \
	 make -j)

test:
	(cd $(BUILD); make test -s)

clean: ; rm -rf $(BUILD)

.PHONY: all build clean
