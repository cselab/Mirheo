INST_TARGETS = make

include ../config/install
include ../make/common.mk

CONFIG = ../config
UTILS  = ../utils
HOST   = `$(UTILS)/host`

make: make.in
	sed "s/@EXE_PREFIX@/$(EXE_PREFIX)/" $< > tmp
	$(UTILS)/insertFile tmp "@MAKE_FLAGS@" $(CONFIG)/make.flags.$(HOST) > $@
	chmod +x $@
	rm -rf tmp

clean:; rm -rf make

.PHONY: clean
