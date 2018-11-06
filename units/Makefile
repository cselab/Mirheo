TARGETS = bounce celllists
all: $(TARGETS)

COMPILE=@(cd $@; mkdir -p build; cd build; cmake ..; make -j)

bounce:    ; $(COMPILE)
celllists: ; $(COMPILE)

clean:
	@for t in $(TARGETS); do     \
	    rm -rf $$t/build;        \
            echo "rm -rf $$t/build"; \
	done

.PHONY: all $(TARGETS) clean
