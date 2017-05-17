
sinclude ../Makefile.tests_common

CLEANCOMMAND+= alfven_wave

#Define this tests level
MY_TEST_LEVEL=1
VAL:= $(shell expr ${MY_TEST_LEVEL} \<= ${TEST_LEVEL})
ifeq "$(VAL)" "0"
all:
	@echo "==> Skipping test in ${PWD}/$(TEST_FOLD) (Test_level ${TEST_LEVEL}, This test ${MY_TEST_LEVEL})."
else
all: alfven_wave 
	$(call RUNCOMMAND,alfven_wave,alfven_wave.in,)
endif

alfven_wave.o:  ${TEST_MODS} 

alfven_wave: alfven_wave.o $(TEST_MODS)
	$(LD) $(LDFLAGS) -o $@ $^ $(LIBS)

