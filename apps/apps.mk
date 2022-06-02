APPS_SRC = $(wildcard apps/*.cc)
APPS = $(patsubst apps/%.cc, $(BUILD_PATH)/apps/%, $(APPS_SRC))

# -ltcmalloc_and_profiler
APPS_LDFLAGS = -Wl,-rpath,$(DEPS_PATH)/lib $(PS_LDFLAGS_SO) -pthread -l boost_system -l boost_program_options -I apps/eigen3/ ${PS_LDFLAGS} -fopenmp

$(BUILD_PATH)/apps/% : apps/%.cc $(BUILD_PATH)/libps.a
	mkdir -p $(BUILD_PATH)/apps
	$(CXX) $(CFLAGS) -MM -MT $(BUILD_PATH)/apps/$* $< >$(BUILD_PATH)/apps/$*.d
	$(CXX) $(CFLAGS) -o $@ $(filter %.cc %.a, $^) $(APPS_LDFLAGS)


-include $(BUILD_PATH)/apps/*.d


# enable "make apps/[app]"
apps/% : apps/%.cc
	$(MAKE) $(BUILD_PATH)/apps/$*
