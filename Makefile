ifdef config
include $(config)
endif

include make/ps.mk

ifndef CXX
CXX = g++
endif

ifndef DEPS_PATH
DEPS_PATH = $(shell pwd)/deps
endif

ifndef BUILD_PATH
BUILD_PATH = build
endif

ifndef PROTOC
PROTOC = ${DEPS_PATH}/bin/protoc
endif

ifndef PROTOC
PROTOC = ${DEPS_PATH}/bin/protoc
endif

ifdef CXX11_ABI
	ABI := -D_GLIBCXX_USE_CXX11_ABI=$(CXX11_ABI)
endif

ifdef KEY_TYPE
ADD_CFLAGS += -DKEY_TYPE=$(KEY_TYPE)
endif



INCPATH = -I./src -I./include -I$(DEPS_PATH)/include
CFLAGS = -std=c++14 -msse2 -fPIC -O3 -ggdb -march=native -Wall -finline-functions $(ABI) $(INCPATH) $(ADD_CFLAGS)

all: ps tests apps

include make/deps.mk

clean:
	rm -rf $(BUILD_PATH)

clean-all: clean
	find src -name "*.pb.[ch]*" -delete
	rm -rf $(DEPS_PATH)
	rm -rf deps/
	rm -rf deps_bindings/
	rm -rf protobuf-*

lint:
	python tests/lint.py ps all include/ps src

ps: $(BUILD_PATH)/libps.a

OBJS = $(addprefix $(BUILD_PATH)/, customer.o postoffice.o van.o meta.pb.o)
$(BUILD_PATH)/libps.a: $(OBJS)
	ar crv $@ $(filter %.o, $?)

$(BUILD_PATH)/%.o: src/%.cc ${ZMQ} src/meta.pb.h
	@mkdir -p $(@D)
	$(CXX) $(INCPATH) -std=c++14 -MM -MT $(BUILD_PATH)/$*.o $< >$(BUILD_PATH)/$*.d
	$(CXX) $(CFLAGS) -c $< -o $@

src/%.pb.cc src/%.pb.h : src/%.proto ${PROTOBUF}
	$(PROTOC) --cpp_out=./src --proto_path=./src $<

-include $(BUILD_PATH)/*.d
-include $(BUILD_PATH)/*/*.d

include tests/test.mk
tests: $(TESTS)

include apps/apps.mk
apps: $(APPS)
