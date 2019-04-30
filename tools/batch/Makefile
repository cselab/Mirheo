INST_TARGETS = batch

CONFIG = ../config

# TODO
BATCH_CMD=bash

include $(CONFIG)/install
include ../make/common.mk

%: %.in $(CONFIG)/install
	@echo create $(EXE_PREFIX).$@
	@sed "s/@EXE_PREFIX@/$(EXE_PREFIX)/" $< | \
	 sed "s/@BATCH_CMD@/$(BATCH_CMD)/" > $@
	@chmod +x $@

test:; atest `find tests -type f`

.PHONY: test
