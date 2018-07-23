INST_TARGETS = load load.falcon \
               load.post load.post.falcon

include ../config
include ../make/common.mk

%: %.in
	sed "s/@EXE_PREFIX@/$(EXE_PREFIX)/" $< > $@
