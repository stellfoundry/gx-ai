
sinclude ../Makefile.tests_common

CLEANCOMMAND+= cyclone_linear

#Define this tests level
MY_TEST_LEVEL=1
VAL:= $(shell expr ${MY_TEST_LEVEL} \<= ${TEST_LEVEL})
ifeq "$(VAL)" "0"
all:
	@echo "==> Skipping test in ${PWD}/$(TEST_FOLD) (Test_level ${TEST_LEVEL}, This test ${MY_TEST_LEVEL})."
else
all: cyclone_linear 
	$(call RUNCOMMAND,cyclone_linear,cyclone_linear.in,)
endif

cyclone_linear.o:  ${TEST_MODS} 

cyclone_linear: cyclone_linear.o $(TEST_MODS)
	$(LD) $(LDFLAGS) -o $@ $^ $(LIBS)

