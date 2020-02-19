mirheo_double=$(shell mir.run -n 1 python -c "import mirheo; u = mirheo.Mirheo((1,1,1), (1,1,1), 0, log_filename='stdout', no_splash=True); u.log_compile_options()" | grep MIRHEO_DOUBLE)

ifeq (,$(findstring ON,$(mirheo_double)))
    atest_args=ATEST_DIR=test_data
else
    atest_args=ATEST_DIR=test_data_double
endif

test:
	$(atest_args) atest `find . \( -name "*.py" -o -name "*.sh" \)`

.PHONY: test test_double
