APPS_SRC = $(wildcard apps/*.cc)
APPS = $(patsubst apps/%.cc, apps/%, $(APPS_SRC))

# -ltcmalloc_and_profiler
APPS_LDFLAGS = -Wl,-rpath,$(DEPS_PATH)/lib $(PS_LDFLAGS_SO) -pthread -l boost_system -l boost_program_options -I third_party/eigen3/ ${LAPSE_EXTERNAL_LDFLAGS}

apps/% : apps/%.cc build/libps.a
	mkdir -p build/apps
	$(CXX) -std=c++0x $(CFLAGS) -MM -MT apps/$* $< >apps/$*.d
	$(CXX) -std=c++0x $(CFLAGS) -o build/$@ $(filter %.cc %.a, $^) $(APPS_LDFLAGS)

-include apps/*.d
