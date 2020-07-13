TESTS_SRC = $(wildcard tests/test_*.cc)
TESTS = $(patsubst tests/test_%.cc, tests/test_%, $(TESTS_SRC))

# -ltcmalloc_and_profiler
TESTS_LDFLAGS = -Wl,-rpath,$(DEPS_PATH)/lib $(PS_LDFLAGS_SO) -pthread -l boost_system -l boost_program_options ${LAPSE_EXTERNAL_LDFLAGS}
tests/% : tests/%.cc build/libps.a
	$(CXX) -std=c++0x $(CFLAGS) -MM -MT tests/$* $< >tests/$*.d
	$(CXX) -std=c++0x $(CFLAGS) -o $@ $(filter %.cc %.a, $^) $(TESTS_LDFLAGS)

-include tests/*.d
