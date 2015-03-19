# include system-dependent make variables
include Makefile.deps
ifndef GK_SYSTEM
	ifdef SYSTEM
$(warning SYSTEM environment variable is obsolete)
$(warning use GK_SYSTEM instead)
	GK_SYSTEM = $(SYSTEM)
	else
$(error GK_SYSTEM is not set)
	endif
endif
include Makefile.depend
include Makefile.$(GK_SYSTEM)
export GK_HEAD_DIR=$(PWD)
TEST_DIR=$(PWD)/tests
include tests/Makefile.tests_and_benchmarks

