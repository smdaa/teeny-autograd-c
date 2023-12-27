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

# Include libpng as a submodule
set(DOXYGEN_SKIP_DOT TRUE)
add_subdirectory(extern/libpng)

# Create the paint executable and link with the library and libpng
add_executable(paint examples/paint/main.c)
target_link_libraries(paint teeny-autograd-c png m)

# Set the build type to Release by default if not specified
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

if(BUILD_TESTS)
    # Include CMocka as a submodule
    add_subdirectory(extern/cmocka)

    # Create test executables and link with cmocka and the library
    add_executable(test_ndarray test/test_ndarray.c)
    add_executable(test_variable test/test_variable.c)
    add_executable(test_multilayer_perceptron test/test_multilayer_perceptron.c)

    target_link_libraries(test_ndarray teeny-autograd-c cmocka m)
    target_link_libraries(test_variable teeny-autograd-c cmocka m)
    target_link_libraries(test_multilayer_perceptron teeny-autograd-c cmocka m)

    # Add a test target to run your tests
    enable_testing()

    # Set the TEST_DATA_DIR environment variable for the tests
    set(TEST_ENVIRONMENT_VARS
            "TEST_NDARRAY_DATA_DIR=${CMAKE_CURRENT_SOURCE_DIR}/test/test_data/test_ndarray"
            "TEST_VARIABLE_DATA_DIR=${CMAKE_CURRENT_SOURCE_DIR}/test/test_data/test_variable"
            "TEST_MULTILAYER_PERCEPTRON_DATA_DIR=${CMAKE_CURRENT_SOURCE_DIR}/test/test_data/test_multilayer_perceptron"
    )

    foreach(test_name IN ITEMS test_ndarray test_variable test_multilayer_perceptron)
        add_test(NAME ${test_name} COMMAND ${test_name})
        set_tests_properties(${test_name} PROPERTIES
                WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
                ENVIRONMENT "${TEST_ENVIRONMENT_VARS}"
        )
    endforeach()
endif()
