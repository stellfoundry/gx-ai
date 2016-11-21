
sinclude ../Makefile.tests_common

CLEANCOMMAND+= miller_cyclone

#Define this tests level
MY_TEST_LEVEL=1
VAL:= $(shell expr ${MY_TEST_LEVEL} \<= ${TEST_LEVEL})
ifeq "$(VAL)" "0"
all:
	@echo "==> Skipping test in ${PWD}/$(TEST_FOLD) (Test_level ${TEST_LEVEL}, This test ${MY_TEST_LEVEL})."
else
all: miller_cyclone 
	$(call RUNCOMMAND,miller_cyclone,miller_cyclone.in,)
endif

miller_cyclone.o:  ${TEST_MODS} 

miller_cyclone: miller_cyclone.o $(TEST_MODS)
	$(LD) $(LDFLAGS) -o $@ $^ $(LIBS)

