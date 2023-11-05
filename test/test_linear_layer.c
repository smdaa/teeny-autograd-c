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
    int batch_size = 4;
    linear_layer *ll = new_linear_layer(in_size, out_size, batch_size);

    assert_non_null(ll);
    assert_int_equal(ll->in_size, in_size);
    assert_int_equal(ll->out_size, out_size);
    assert_int_equal(ll->weights->val->dim, 2);
    assert_int_equal(ll->weights->val->size, in_size * out_size);
    assert_int_equal(ll->weights->grad->dim, 2);
    assert_int_equal(ll->weights->grad->size, in_size * out_size);
    assert_int_equal(ll->bias->val->dim, 2);
    assert_int_equal(ll->bias->val->size, out_size);
    assert_int_equal(ll->bias->grad->dim, 2);
    assert_int_equal(ll->bias->grad->size, out_size);

    free_linear_layer(ll);
}

static void test_forward_linear_layer(void **state)
{
    (void)state;
    int in_size = 128;
    int out_size = 256;
    int batch_size = 32;
    linear_layer *ll = new_linear_layer(in_size, out_size, batch_size);

    variable *input = new_variable(read_ndarray("./test_data/test_forward_linear_layer_input_val.txt"));
    ndarray *output_val_gt = read_ndarray("./test_data/test_forward_linear_layer_output_val.txt");
    variable *weights = new_variable(read_ndarray("./test_data/test_forward_linear_layer_weights_val.txt"));
    variable *bias = new_variable(read_ndarray("./test_data/test_forward_linear_layer_bias_val.txt"));
    ndarray *up_stream_grad = read_ndarray("./test_data/test_forward_linear_layer_up_stream_grad.txt");
    ndarray *weights_grad_gt = read_ndarray("./test_data/test_forward_linear_layer_weights_grad.txt");
    ndarray *bias_grad_gt = read_ndarray("./test_data/test_forward_linear_layer_bias_grad.txt");

    assert_int_equal(weights->val->dim, 2);
    assert_int_equal(weights->val->shape[0], in_size);
    assert_int_equal(weights->val->shape[1], out_size);
    assert_int_equal(bias->val->dim, 2);
    assert_int_equal(bias->val->shape[0], 1);
    assert_int_equal(bias->val->shape[1], out_size);

    free(ll->weights);
    ll->weights = weights;
    free(ll->bias);
    ll->bias = bias;

    variable *output = forward_linear_layer(ll, input);
    free(output->grad);
    output->grad = up_stream_grad;
    backward(output);

    assert_true(is_equal(output->val, output_val_gt));
    assert_true(is_equal(ll->weights->grad, weights_grad_gt));
    assert_true(is_equal(ll->bias->grad, bias_grad_gt));

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