
sinclude ../Makefile.tests_common

CLEANCOMMAND+= test_read_namelist

TEST_MODS = $(GK_HEAD_DIR)/read_namelist.o $(GK_HEAD_DIR)/global_variables.o $(GK_HEAD_DIR)/unit_tests.o

#Define this tests level
MY_TEST_LEVEL=1
VAL:= $(shell expr ${MY_TEST_LEVEL} \<= ${TEST_LEVEL})
ifeq "$(VAL)" "0"
all:
	@echo "==> Skipping test in ${PWD}/$(TEST_FOLD) (Test_level ${TEST_LEVEL}, This test ${MY_TEST_LEVEL})."
else
all: test_read_namelist 
	$(call RUNCOMMAND,test_read_namelist,test_read_namelist.in,)
endif

test_read_namelist.o:  ${TEST_MODS} 

test_read_namelist: test_read_namelist.o $(TEST_MODS)
	$(LD) $(LDFLAGS) -o $@ $^ $(LIBS)

