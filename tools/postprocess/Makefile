INST_TARGETS = avgh5 avgh5.py post

CONFIG = ../config

include $(CONFIG)/install
include ../make/common.mk

%: %.in $(CONFIG)/install
	@echo create $(EXE_PREFIX).$@
	@sed "s/@EXE_PREFIX@/$(EXE_PREFIX)/" $< > $@
	@chmod +x $@

test:; @atest `find test -type f`

.PHONY: install test
