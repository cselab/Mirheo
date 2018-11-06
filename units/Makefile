NOT_COMPILING = bounce halo manyranks wall
TARGETS = celllists interaction pid 
all: $(TARGETS)

COMPILE=@(cd $@; mkdir -p build; cd build; cmake ..; make -j)

bounce:            ; $(COMPILE)
celllists:         ; $(COMPILE)
halo:              ; $(COMPILE)
interaction:       ; $(COMPILE)
manyranks:         ; $(COMPILE)
mesh_bounce:       ; $(COMPILE)
mesh_bounce_full:  ; $(COMPILE)
mesh_leak:         ; $(COMPILE)
obj_halo:          ; $(COMPILE)
one_rank:          ; $(COMPILE)
pid:               ; $(COMPILE)
redist:            ; $(COMPILE)
rigid:             ; $(COMPILE)
scheduler:         ; $(COMPILE)
serializer:        ; $(COMPILE)
wall:              ; $(COMPILE)

clean:
	@for t in $(TARGETS); do     \
	    rm -rf $$t/build;        \
            echo "rm -rf $$t/build"; \
	done

.PHONY: all $(TARGETS) $(NOT_COMPILING) clean
