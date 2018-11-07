all: build

BUILD=build

build:
	(mkdir -p $(BUILD); \
	 cd $(BUILD); cmake ..; \
	  make -j)

test: build
	(cd $(BUILD); make test)

clean: ; rm -rf $(BUILD)

.PHONY: all build clean
