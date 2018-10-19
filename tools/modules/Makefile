INST_TARGETS = load load.post

include ../config/install
include ../make/common.mk

CONFIG = ../config
UTILS  = ../utils
HOST   = `$(UTILS)/host`

%: %.in
	$(UTILS)/insertFile $< "@MODULES@" $(CONFIG)/$@.$(HOST) > $@

clean:; rm -rf $(INST_TARGETS)

.PHONY: clean
