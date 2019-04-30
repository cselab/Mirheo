INST_TARGETS = run

CONFIG = ../config

include $(CONFIG)/install
include ../make/common.mk

%: %.in $(CONFIG)/install
	@echo create $(EXE_PREFIX).$@
	@sed "s/@EXE_PREFIX@/$(EXE_PREFIX)/" $< | \
	 sed "s/@RUN_CMD@/$(RUN_CMD)/" > $@
	@chmod +x $@

test:; atest `find tests -type f`

.PHONY: test
