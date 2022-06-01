TESTS_SRC = $(wildcard tests/test_*.cc)
TESTS = $(patsubst tests/test_%.cc, tests/test_%, $(TESTS_SRC))

# -ltcmalloc_and_profiler
TESTS_LDFLAGS = -Wl,-rpath,$(DEPS_PATH)/lib $(PS_LDFLAGS_SO) -pthread -l boost_system -l boost_program_options ${LAPSE_EXTERNAL_LDFLAGS}

$(BUILD_PATH)/tests/% : tests/%.cc $(BUILD_PATH)/libps.a
	mkdir -p $(BUILD_PATH)/tests
	$(CXX) $(CFLAGS) -MM -MT $(BUILD_PATH)/tests/$* $< >$(BUILD_PATH)/tests/$*.d
	$(CXX) $(CFLAGS) -o $@ $(filter %.cc %.a, $^) $(TESTS_LDFLAGS)

-include $(BUILD_PATH)/tests/*.d

# enable "make tests/[test]"
tests/% : tests/%.cc
	$(MAKE) $(BUILD_PATH)/tests/$*
