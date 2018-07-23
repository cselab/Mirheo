include ../config

src = avgh5 avgh5.py

install: $(src)
	@echo install $^ in $(BIN_PREFIX)
	@mkdir -p $(BIN_PREFIX)
	@for f in $(src); do                          \
           cp "$$f" "$(BIN_PREFIX)/$(EXE_PREFIX).$$f"; \
	done

avgh5: avgh5.in
	sed "s/@EXE_PREFIX@/$(EXE_PREFIX)/" $< > $@
	chmod +x $@

test:
	@atest `find test -type f`

.PHONY: install test
