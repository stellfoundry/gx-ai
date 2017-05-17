
sinclude ../Makefile.tests_common

CLEANCOMMAND+= theta_grid

#Define this tests level
MY_TEST_LEVEL=1
VAL:= $(shell expr ${MY_TEST_LEVEL} \<= ${TEST_LEVEL})
ifeq "$(VAL)" "0"
all:
	@echo "==> Skipping test in ${PWD}/$(TEST_FOLD) (Test_level ${TEST_LEVEL}, This test ${MY_TEST_LEVEL})."
else
all: theta_grid 
	$(call RUNCOMMAND,theta_grid,theta_grid.in,)
endif

theta_grid.o:  ${TEST_MODS} 

theta_grid: theta_grid.o $(TEST_MODS)
	$(LD) $(LDFLAGS) -o $@ $^ $(LIBS)

