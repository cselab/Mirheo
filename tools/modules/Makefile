INST_TARGETS = load load.post \
	       load.falcon load.post.falcon \
	       load.panda  load.post.panda  \
	       load.daint  load.post.daint  \
		   load.__default load.post.__default

include ../config
include ../make/common.mk

%: %.in
	sed "s/@EXE_PREFIX@/$(EXE_PREFIX)/" $< > $@
