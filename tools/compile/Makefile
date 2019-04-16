INST_TARGETS = make

CONFIG = ../config
UTILS  = ../utils

include $(CONFIG)/install
include ../make/common.mk

HOST := $(shell $(UTILS)/host)

MAKEFLAGS_FILE := $(CONFIG)/make.flags.$(HOST)

make: make.in $(MAKEFLAGS_FILE) $(CONFIG)/install
	@echo create $(EXE_PREFIX).$@
	@sed "s/@EXE_PREFIX@/$(EXE_PREFIX)/" $< > tmp
	@$(UTILS)/insertFile tmp "@MAKE_FLAGS@" $(MAKEFLAGS_FILE) > $@
	@chmod +x $@
	@rm -rf tmp

clean:; rm -rf make

.PHONY: clean
