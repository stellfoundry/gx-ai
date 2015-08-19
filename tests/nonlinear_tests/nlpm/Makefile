
sinclude ../Makefile.tests_common

CLEANCOMMAND+= nlpm

#Define this tests level
MY_TEST_LEVEL=1
VAL:= $(shell expr ${MY_TEST_LEVEL} \<= ${TEST_LEVEL})
ifeq "$(VAL)" "0"
all:
	@echo "==> Skipping test in ${PWD}/$(TEST_FOLD) (Test_level ${TEST_LEVEL}, This test ${MY_TEST_LEVEL})."
else
all: nlpm 
	#$(call RUNCOMMAND,../../../gryfx,nlpm_setup_phix.in,)
	#$(call RUNCOMMAND,nlpm,nlpm_phix_dens.in,)
	##$(call RUNCOMMAND,nlpm,nlpm_phix_tpar.in,)
	##$(call RUNCOMMAND,nlpm,nlpm_phix_tprp.in,)
	#$(call RUNCOMMAND,nlpm,nlpm_phix_odd.in,)
	#$(call RUNCOMMAND,../../../gryfx,nlpm_setup_phiy.in,)
	#$(call RUNCOMMAND,nlpm,nlpm_phiy_dens.in,)
	#$(call RUNCOMMAND,nlpm,nlpm_phiy_tpar.in,)
	#$(call RUNCOMMAND,nlpm,nlpm_phiy_tprp.in,)
	#$(call RUNCOMMAND,nlpm,nlpm_phiy_odd.in,)
endif

nlpm.o:  ${TEST_MODS} 

nlpm: nlpm.o $(TEST_MODS)
	$(LD) $(LDFLAGS) -o $@ $^ $(LIBS)

