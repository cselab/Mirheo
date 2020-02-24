mirheo_double=$(shell python -m mirheo compile_opt useDouble)

ifeq (,$(findstring ON,$(mirheo_double)))
    atest_args=ATEST_DIR=test_data
else
    atest_args=ATEST_DIR=test_data_double
endif

test:
	$(atest_args) atest `find . \( -name "*.py" -o -name "*.sh" \)`

.PHONY: test test_double
