INST_TARGETS = load load.post

CONFIG = ../config
UTILS  = ../utils

include $(CONFIG)/install
include ../make/common.mk

HOST := $(shell $(UTILS)/host)

%: %.in $(CONFIG)/%.$(HOST) $(CONFIG)/install
	@echo create $(EXE_PREFIX).$@
	@$(UTILS)/insertFile $< "@MODULES@" $(CONFIG)/$@.$(HOST) > $@

clean:; rm -rf $(INST_TARGETS)

.PHONY: clean
