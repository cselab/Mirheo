INST_TARGETS = restart.id

CONFIG = ../config

include $(CONFIG)/install
include ../make/common.mk

%: %.in $(CONFIG)/install
	@echo create $(EXE_PREFIX).$@
	@sed "s/@EXE_PREFIX@/$(EXE_PREFIX)/" $< > $@
	@chmod +x $@

test:; @atest tests/main

.PHONY: test
