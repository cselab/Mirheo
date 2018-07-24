INST_TARGETS = load load.post \
	       load.falcon load.post.falcon \
	       load.daint load.post.daint

include ../config
include ../make/common.mk

%: %.in
	sed "s/@EXE_PREFIX@/$(EXE_PREFIX)/" $< > $@
