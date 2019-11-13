all: build

BUILD=build

LOCAL_CMAKE_FLAGS = "-Dgtest_disable_pthreads=ON"
LOCAL_CMAKE_FLAGS += "-DBUILD_GMOCK=OFF"
LOCAL_CMAKE_FLAGS += "-DINSTALL_GTEST=OFF"

# Detect "-j XY" pattern. Does not detect "-j".
# https://stackoverflow.com/a/33616144
MAKE_PID := $(shell echo $$PPID)
JOBS := $(shell ps T | sed -n 's/\s*$(MAKE_PID) .*$(MAKE).* \(-j\|--jobs\) *\([0-9][0-9]*\).*/\2/p')


build:
	(mkdir -p $(BUILD) && \
	 cd $(BUILD) && cmake ${CMAKE_FLAGS} $(LOCAL_CMAKE_FLAGS) .. && \
	 $(MAKE))

test:
	(cd $(BUILD) && ctest -s -j${JOBS})

clean:
	rm -rf $(BUILD)

.PHONY: all build clean
