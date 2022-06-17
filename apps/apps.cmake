set(APPS matrix_factorization knowledge_graph_embeddings word2vec simple)


add_executable(matrix_factorization ${PROJECT_SOURCE_DIR}/apps/matrix_factorization.cc)
target_include_directories(matrix_factorization PUBLIC ${INCS})
target_link_libraries(matrix_factorization
  PUBLIC pthread
  PUBLIC adaps
  PUBLIC Eigen3::Eigen
  PUBLIC Boost::system
  PUBLIC Boost::program_options
)

find_package(OpenMP)
add_executable(knowledge_graph_embeddings ${PROJECT_SOURCE_DIR}/apps/knowledge_graph_embeddings.cc)
target_include_directories(knowledge_graph_embeddings PUBLIC ${INCS})
target_link_libraries(knowledge_graph_embeddings 
  PUBLIC pthread
  PUBLIC adaps
  PUBLIC Boost::system
  PUBLIC Boost::program_options
  PUBLIC OpenMP::OpenMP_CXX
)

add_executable(word2vec ${PROJECT_SOURCE_DIR}/apps/word2vec.cc)
target_include_directories(word2vec PUBLIC ${INCS})
target_link_libraries(word2vec
  PUBLIC pthread
  PUBLIC adaps
  PUBLIC Boost::system
  PUBLIC Boost::program_options
)

add_executable(simple ${PROJECT_SOURCE_DIR}/apps/simple.cc)
target_include_directories(simple PUBLIC ${INCS})
target_link_libraries(simple
  PUBLIC pthread
  PUBLIC adaps
  PUBLIC Boost::system
  PUBLIC Boost::program_options
)

set_target_properties(${APPS}
  PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/apps"
)

add_custom_target(apps)
add_dependencies(apps ${APPS})
