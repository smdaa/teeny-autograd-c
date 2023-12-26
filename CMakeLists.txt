cmake_minimum_required(VERSION 3.26)
project(teeny-autograd-c C)

set(CMAKE_C_STANDARD 11)

# Collect source files for the library
set(LIB_SRC_FILES
        src/ndarray.c
        src/variable.c
        src/multilayer_perceptron.c
)

# Create the library
add_library(teeny-autograd-c ${LIB_SRC_FILES})

# Option to build tests (default is ON)
option(BUILD_TESTS "Build tests" ON)

if(BUILD_TESTS)
    # Include CMocka as a submodule
    add_subdirectory(external/cmocka)

    # Create test executable and link with cmocka and the library
    add_executable(test_ndarray test/test_ndarray.c)
    add_executable(test_variable test/test_variable.c)
    add_executable(test_multilayer_perceptron test/test_multilayer_perceptron.c)

    target_link_libraries(test_ndarray teeny-autograd-c cmocka m)
    target_link_libraries(test_variable teeny-autograd-c cmocka m)
    target_link_libraries(test_multilayer_perceptron teeny-autograd-c cmocka m)

    # Add a test target to run your tests
    enable_testing()

    # Set the TEST_DATA_DIR environment variable for the test
    add_test(NAME test_ndarray COMMAND test_ndarray)
    set_tests_properties(test_ndarray PROPERTIES
            WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
            ENVIRONMENT "TEST_NDARRAY_DATA_DIR=${CMAKE_CURRENT_SOURCE_DIR}/test/test_data/test_ndarray"
    )

    add_test(NAME test_variable COMMAND test_variable)
    set_tests_properties(test_variable PROPERTIES
            WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
            ENVIRONMENT "TEST_VARIABLE_DATA_DIR=${CMAKE_CURRENT_SOURCE_DIR}/test/test_data/test_variable"
    )

    add_test(NAME test_multilayer_perceptron COMMAND test_multilayer_perceptron)
    set_tests_properties(test_multilayer_perceptron PROPERTIES
            WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
            ENVIRONMENT "TEST_MULTILAYER_PERCEPTRON_DATA_DIR=${CMAKE_CURRENT_SOURCE_DIR}/test/test_data/test_multilayer_perceptron"
    )
endif()
