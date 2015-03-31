
sinclude ../Makefile.tests_common

CLEANCOMMAND+= miller_linear

#Define this tests level
MY_TEST_LEVEL=1
VAL:= $(shell expr ${MY_TEST_LEVEL} \<= ${TEST_LEVEL})
ifeq "$(VAL)" "0"
all:
	@echo "==> Skipping test in ${PWD}/$(TEST_FOLD) (Test_level ${TEST_LEVEL}, This test ${MY_TEST_LEVEL})."
else
all: miller_linear 
	$(call RUNCOMMAND,miller_linear,miller_linear.in,)
endif

miller_linear.o:  ${TEST_MODS} 

miller_linear: miller_linear.o $(TEST_MODS)
	$(LD) $(LDFLAGS) -o $@ $^ $(LIBS)

