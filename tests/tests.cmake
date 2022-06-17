set(TESTS test_dynamic_allocation test_locality_api test_many_key_operations test_sampling test_set_operation)

foreach( test ${TESTS} )
  add_executable( ${test} ${PROJECT_SOURCE_DIR}/tests/${test}.cc )
  target_include_directories(${test} PUBLIC ${INCS})
  target_link_libraries(${test}
    PUBLIC pthread
    PUBLIC adaps
    PUBLIC Boost::system
    PUBLIC Boost::program_options
  )
endforeach()

set_target_properties(${TESTS}
  PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/tests"
)

add_custom_target(tests)
add_dependencies(tests ${TESTS})
