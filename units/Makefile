all: build

BUILD=build

build:
	(mkdir -p $(BUILD) && \
	 cd $(BUILD) && cmake .. -DGTEST_HAS_PTHREAD=0 && \
	 make -j)

test:
	(cd $(BUILD); make test)

clean: ; rm -rf $(BUILD)

.PHONY: all build clean
