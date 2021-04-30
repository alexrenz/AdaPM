#---------------------------------------------------------------------------------------
#  parameter server configuration script
#
#  include ps.mk after the variables are set
#
#----------------------------------------------------------------------------------------

PS_LDFLAGS_SO = -L$(DEPS_PATH)/lib -lprotobuf-lite -lzmq
PS_LDFLAGS_A = $(addprefix $(DEPS_PATH)/lib/, libprotobuf-lite.a libzmq.a)
