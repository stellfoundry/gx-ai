
sinclude ../Makefile.tests_common

CLEANCOMMAND+= test_geometry

#TEST_MODS = $(GK_HEAD_DIR)/geometry.o $(GK_HEAD_DIR)/global_variables.o $(GK_HEAD_DIR)/unit_tests.o

#Define this tests level
MY_TEST_LEVEL=1
VAL:= $(shell expr ${MY_TEST_LEVEL} \<= ${TEST_LEVEL})
ifeq "$(VAL)" "0"
all:
	@echo "==> Skipping test in ${PWD}/$(TEST_FOLD) (Test_level ${TEST_LEVEL}, This test ${MY_TEST_LEVEL})."
else
all: test_geometry 
	$(call RUNCOMMAND,test_geometry,test_geometry.in,)
endif

test_geometry.o:  ${TEST_MODS} 

test_geometry: test_geometry.o $(TEST_MODS)
	$(LD) $(LDFLAGS) -o $@ $^ $(LIBS)

