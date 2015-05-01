
sinclude ../Makefile.tests_common

CLEANCOMMAND+= cyclone

#Define this tests level
MY_TEST_LEVEL=2
VAL:= $(shell expr ${MY_TEST_LEVEL} \<= ${TEST_LEVEL})
ifeq "$(VAL)" "0"
all:
	@echo "==> Skipping test in ${PWD}/$(TEST_FOLD) (Test_level ${TEST_LEVEL}, This test ${MY_TEST_LEVEL})."
else
all: cyclone 
	$(call RUNCOMMAND,cyclone,cyclone.in,)
endif

cyclone.o:  ${TEST_MODS} 

cyclone: cyclone.o $(TEST_MODS)
	$(LD) $(LDFLAGS) -o $@ $^ $(LIBS)

