find_package(GTest REQUIRED)
include(GoogleTest)

add_executable(tilelang_cpp_tests
  sunmmio_tile_loop_fusion_analysis_test.cc
  sunmmio_tile_loop_fusion_cost_model_test.cc
  sunmmio_tile_loop_fusion_planner_internal_test.cc
  sunmmio_tile_loop_fusion_planner_test.cc
)

set_target_properties(tilelang_cpp_tests PROPERTIES
  CXX_STANDARD 17
  CXX_STANDARD_REQUIRED ON
  CXX_EXTENSIONS OFF
  ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
)

target_include_directories(tilelang_cpp_tests PRIVATE
  ${TILE_LANG_INCLUDES}
  ${PROJECT_SOURCE_DIR}/src
  ${PROJECT_SOURCE_DIR}/src/analysis
  ${PROJECT_SOURCE_DIR}/src/layout
  ${PROJECT_SOURCE_DIR}/src/op
  ${PROJECT_SOURCE_DIR}/src/target
  ${PROJECT_SOURCE_DIR}/src/tileview
  ${PROJECT_SOURCE_DIR}/src/transform
)

target_link_libraries(tilelang_cpp_tests PRIVATE
  tilelang
  GTest::gtest
  GTest::gtest_main
)

gtest_discover_tests(
  tilelang_cpp_tests
  WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
  TEST_DISCOVERY_TIMEOUT 600
  DISCOVERY_MODE POST_BUILD
  PROPERTIES VS_DEBUGGER_WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}"
)
