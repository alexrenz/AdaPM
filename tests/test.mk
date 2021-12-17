TESTS_SRC = $(wildcard tests/test_*.cc)
TESTS = $(patsubst tests/test_%.cc, tests/test_%, $(TESTS_SRC))

# -ltcmalloc_and_profiler
TESTS_LDFLAGS = -Wl,-rpath,$(DEPS_PATH)/lib $(PS_LDFLAGS_SO) -pthread -l boost_system -l boost_program_options ${PS_EXTERNAL_LDFLAGS}

build/tests/% : tests/%.cc build/libps.a
	mkdir -p build/tests
	$(CXX) -std=c++0x $(CFLAGS) -MM -MT build/tests/$* $< >build/tests/$*.d
	$(CXX) -std=c++0x $(CFLAGS) -o $@ $(filter %.cc %.a, $^) $(TESTS_LDFLAGS)

-include build/tests/*.d

# enable "make tests/[test]"
tests/% : tests/%.cc
	make build/tests/$*
