NOT_COMPILING = bounce halo manyranks mesh_bounce mesh_leak obj_halo onerank redist rigid wall
TARGETS = celllists interaction pid scheduler serializer
all: $(TARGETS)

COMPILE=@(cd $@; mkdir -p build; cd build; cmake ..; make -j)

bounce:            ; $(COMPILE)
celllists:         ; $(COMPILE)
halo:              ; $(COMPILE)
interaction:       ; $(COMPILE)
manyranks:         ; $(COMPILE)
mesh_bounce:       ; $(COMPILE)
mesh_leak:         ; $(COMPILE)
obj_halo:          ; $(COMPILE)
onerank:           ; $(COMPILE)
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
