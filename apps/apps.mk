APPS_SRC = $(wildcard apps/*.cc)
APPS = $(patsubst apps/%.cc, build/apps/%, $(APPS_SRC))

# -ltcmalloc_and_profiler
APPS_LDFLAGS = -Wl,-rpath,$(DEPS_PATH)/lib $(PS_LDFLAGS_SO) -pthread -l boost_system -l boost_program_options -I third_party/eigen3/ ${PS_EXTERNAL_LDFLAGS} -fopenmp

build/apps/% : apps/%.cc build/libps.a
	mkdir -p build/apps
	$(CXX) -std=c++0x $(CFLAGS) -MM -MT build/apps/$* $< >build/apps/$*.d
	$(CXX) -std=c++0x $(CFLAGS) -o $@ $(filter %.cc %.a, $^) $(APPS_LDFLAGS)


-include build/apps/*.d


# enable "make apps/[app]"
apps/% : apps/%.cc
	make build/apps/$*
