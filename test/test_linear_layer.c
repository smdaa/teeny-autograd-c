#include <stdlib.h>
#include <stdarg.h>
#include <stddef.h>
#include <setjmp.h>
#include <cmocka.h>
#include <math.h>
#include "../src/linear_layer.h"

#define EPSILON 0.0001

static void test_new_linear_layer(void **state)
{
    (void)state;

    int in_size = 8;
    int out_size = 16;
    linear_layer *ll = new_linear_layer(in_size, out_size);

    assert_non_null(ll);
    assert_int_equal(ll->in_size, in_size);
    assert_int_equal(ll->out_size, out_size);
    assert_int_equal(ll->weights->val->dim, 2);
    assert_int_equal(ll->weights->val->size, in_size * out_size);
    assert_int_equal(ll->weights->grad->dim, 2);
    assert_int_equal(ll->weights->grad->size, in_size * out_size);
    assert_int_equal(ll->bias->val->dim, 1);
    assert_int_equal(ll->bias->val->size, out_size);
    assert_int_equal(ll->bias->grad->dim, 1);
    assert_int_equal(ll->bias->grad->size, out_size);

    free_linear_layer(ll);
}

static void test_forward_linear_layer(void **state)
{
    (void)state;

    int in_size = 128;
    int out_size = 256;
    linear_layer *ll = new_linear_layer(in_size, out_size);

    variable *input = new_variable(read_ndarray("./test_data/test_forward_linear_layer_input_val.txt"));
    ndarray *output_val_gt = read_ndarray("./test_data/test_forward_linear_layer_output_val.txt");
    variable *weights = new_variable(read_ndarray("./test_data/test_forward_linear_layer_weights_val.txt"));
    variable *bias = new_variable(read_ndarray("./test_data/test_forward_linear_layer_bias_val.txt"));

    free(ll->weights);
    ll->weights = weights;
    free(ll->bias);
    ll->bias = bias;

    variable *output = forward_linear_layer(ll, input);
    assert_true(is_equal(output->val, output_val_gt));

    free_linear_layer(ll);
}

int main(void)
{
    const struct CMUnitTest tests[] = {
        cmocka_unit_test(test_new_linear_layer),
        cmocka_unit_test(test_forward_linear_layer),
    };
    return cmocka_run_group_tests(tests, NULL, NULL);
}