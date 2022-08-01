cmake_minimum_required(VERSION 3.10)

project(adaps VERSION 0.1 LANGUAGES CXX)
list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake")

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native -Wall")
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Configuration options
set(PS_BACKEND "14" CACHE STRING "Which data structure to use for the PS backend. '14': vector (default), '1': hash map")
option(PS_TRACE_KEYS "Activate key tracing?" OFF)
option(PS_LOCALITY_STATS "Write out detailed locality stats?" OFF)
set(PS_KEY_TYPE "uint64_t" CACHE STRING "Which data type to use for PS keys? (default: uint64_t")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DPS_BACKEND=${PS_BACKEND} -DPS_TRACE_KEYS=${PS_TRACE_KEYS} -DPS_LOCALITY_STATS=${PS_LOCALITY_STATS} -DPS_KEY_TYPE=${PS_KEY_TYPE}" )

# Default build type: Release
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING
    "Build type. Options: Debug Release RelWithDebInfo MinSizeRel." FORCE)
endif()
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

# libadaps
set(SOURCES src/customer.cc src/postoffice.cc src/van.cc)
add_library(adaps ${SOURCES})
target_include_directories(adaps
  PUBLIC "${PROJECT_BINARY_DIR}"
  PUBLIC "${PROJECT_SOURCE_DIR}/include/"
)
add_library(Threads::Threads INTERFACE IMPORTED)
target_link_libraries(adaps PUBLIC pthread)

# Protobuf
if(DEFINED PROTOBUF_PATH) # use a custom Protobuf
  set(protobuf_MODULE_COMPATIBLE ON CACHE BOOL "")
  find_package(Protobuf 3.0.0 REQUIRED NO_DEFAULT_PATH PATHS ${PROTOBUF_PATH})
else() # use any (likely system-provided) Protobuf
  find_package(Protobuf 3.0.0 REQUIRED)
endif()
set(PROTO_FILES src/meta.proto)
protobuf_generate_cpp(PROTO_SRCS PROTO_HDRS ${PROTO_FILES})
set_property(TARGET adaps PROPERTY SOURCES ${SOURCES} ${PROTO_SRCS} ${PROTO_HDRS})
target_include_directories(adaps PUBLIC ${PROTOBUF_INCLUDE_DIRS})
target_link_libraries(adaps PUBLIC ${PROTOBUF_LIBRARY})

# ZMQ
find_package (ZMQ REQUIRED)
target_include_directories(adaps PUBLIC ${ZMQ_INCLUDE_DIRS})
target_link_libraries(adaps PUBLIC ${ZMQ_LIBRARIES})

# Boost
find_package(Boost 1.63.0 COMPONENTS system program_options iostreams filesystem REQUIRED)

# Eigen
find_package(Eigen3 3.3 REQUIRED)

set(INCS "${PROJECT_BINARY_DIR}" "${PROJECT_SOURCE_DIR}/include/" "${EIGEN3_INCLUDE_DIR}" "${Boost_INCLUDE_DIRS}")

include("${PROJECT_SOURCE_DIR}/apps/apps.cmake")
include("${PROJECT_SOURCE_DIR}/tests/tests.cmake")
