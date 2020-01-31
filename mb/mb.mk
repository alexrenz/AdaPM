MB_SRC = $(wildcard mb/*.cc)
MB = $(patsubst mb/%.cc, mb/%, $(MB_SRC))

# -ltcmalloc_and_profiler
MB_LDFLAGS = -Wl,-rpath,$(DEPS_PATH)/lib $(PS_LDFLAGS_SO) -pthread -l boost_system -l boost_program_options -I third_party/eigen3/ ${LAPSE_EXTERNAL_LDFLAGS}

mb/% : mb/%.cc build/libps.a
	mkdir -p build/mb
	$(CXX) -std=c++0x $(CFLAGS) -MM -MT mb/$* $< >mb/$*.d
	$(CXX) -std=c++0x $(CFLAGS) -o build/$@ $(filter %.cc %.a, $^) $(MB_LDFLAGS)

-include mb/*.d
