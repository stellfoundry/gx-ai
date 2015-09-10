
sinclude ../Makefile.tests_common

CLEANCOMMAND+= cyclone_linear_multispec

#Define this tests level
MY_TEST_LEVEL=2
VAL:= $(shell expr ${MY_TEST_LEVEL} \<= ${TEST_LEVEL})
ifeq "$(VAL)" "0"
all:
	@echo "==> Skipping test in ${PWD}/$(TEST_FOLD) (Test_level ${TEST_LEVEL}, This test ${MY_TEST_LEVEL})."
else
all: cyclone_linear_multispec 
#	$(call RUNCOMMAND,cyclone_linear_multispec,cyclone_linear_2spec.in,)
	$(call RUNCOMMAND,cyclone_linear_multispec,cyclone_linear_3spec.in,)
endif

cyclone_linear_multispec.o:  ${TEST_MODS} 

cyclone_linear_multispec: cyclone_linear_multispec.o $(TEST_MODS)
	$(LD) $(LDFLAGS) -o $@ $^ $(LIBS)

