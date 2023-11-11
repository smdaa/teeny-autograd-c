#include <stdlib.h>
#include <stdarg.h>
#include <stddef.h>
#include <setjmp.h>
#include <cmocka.h>
#include <math.h>
#include "../src/multilayer_perceptron.h"

static void test_new_multilayer_perceptron(void **state)
{
    (void)state;

    multilayer_perceptron *mlp = new_multilayer_perceptron(2, 64, (int[]){16, 32}, (int[]){32, 8}, (activation_function[]){LINEAR, LINEAR});
    assert_non_null(mlp);
    assert_int_equal(mlp->n_layers, 2);
    assert_int_equal(mlp->batch_size, 64);
    assert_int_equal(mlp->in_sizes[0], 16);
    assert_int_equal(mlp->in_sizes[1], 32);
    assert_int_equal(mlp->out_sizes[0], 32);
    assert_int_equal(mlp->out_sizes[1], 8);
    assert_non_null(mlp->weights);
    assert_non_null(mlp->bias);
    assert_non_null(mlp->weights[0]);
    assert_non_null(mlp->weights[1]);
    assert_non_null(mlp->bias[0]);
    assert_non_null(mlp->bias[1]);
    assert_int_equal(mlp->weights[0]->val->dim, 2);
    assert_int_equal(mlp->weights[1]->val->dim, 2);
    assert_int_equal(mlp->bias[0]->val->dim, 2);
    assert_int_equal(mlp->bias[1]->val->dim, 2);
    assert_int_equal(mlp->weights[0]->val->size, 16 * 32);
    assert_int_equal(mlp->weights[1]->val->size, 32 * 8);
    assert_int_equal(mlp->bias[0]->val->size, 32);
    assert_int_equal(mlp->bias[1]->val->size, 8);

    free_variable(&(mlp->weights[0]));
    free_variable(&(mlp->weights[1]));
    free_variable(&(mlp->bias[0]));
    free_variable(&(mlp->bias[1]));
    free_multilayer_perceptron(&mlp);
}

static void test_forward_multilayer_perceptron(void **state)
{
    (void)state;

    ndarray *input_val = read_ndarray("./test_data/test_forward_multilayer_perceptron_input_val.txt");
    ndarray *output_val_gt = read_ndarray("./test_data/test_forward_multilayer_perceptron_output_val.txt");
    ndarray *weights0_val = read_ndarray("./test_data/test_forward_multilayer_perceptron_weights0_val.txt");
    ndarray *weights1_val = read_ndarray("./test_data/test_forward_multilayer_perceptron_weights1_val.txt");
    ndarray *weights2_val = read_ndarray("./test_data/test_forward_multilayer_perceptron_weights2_val.txt");
    ndarray *bias0_val = read_ndarray("./test_data/test_forward_multilayer_perceptron_bias0_val.txt");
    ndarray *bias1_val = read_ndarray("./test_data/test_forward_multilayer_perceptron_bias1_val.txt");
    ndarray *bias2_val = read_ndarray("./test_data/test_forward_multilayer_perceptron_bias2_val.txt");
    ndarray *up_stream_grad = read_ndarray("./test_data/test_forward_multilayer_perceptron_up_stream_grad.txt");
    ndarray *weights0_grad_gt = read_ndarray("./test_data/test_forward_multilayer_perceptron_weights0_grad.txt");
    ndarray *weights1_grad_gt = read_ndarray("./test_data/test_forward_multilayer_perceptron_weights1_grad.txt");
    ndarray *weights2_grad_gt = read_ndarray("./test_data/test_forward_multilayer_perceptron_weights2_grad.txt");
    ndarray *bias0_grad_gt = read_ndarray("./test_data/test_forward_multilayer_perceptron_bias0_grad.txt");
    ndarray *bias1_grad_gt = read_ndarray("./test_data/test_forward_multilayer_perceptron_bias1_grad.txt");
    ndarray *bias2_grad_gt = read_ndarray("./test_data/test_forward_multilayer_perceptron_bias2_grad.txt");

    variable *input = new_variable(input_val);
    variable *weights0 = new_variable(weights0_val);
    variable *weights1 = new_variable(weights1_val);
    variable *weights2 = new_variable(weights2_val);
    variable *bias0 = new_variable(bias0_val);
    variable *bias1 = new_variable(bias1_val);
    variable *bias2 = new_variable(bias2_val);

    multilayer_perceptron *mlp = new_multilayer_perceptron(3, 64, (int[]){32, 128, 256}, (int[]){128, 256, 512}, (activation_function[]){SIGMOID, RELU, LINEAR});

    free_variable(&(mlp->weights[0]));
    free_variable(&(mlp->weights[1]));
    free_variable(&(mlp->weights[2]));
    mlp->weights[0] = weights0;
    mlp->weights[1] = weights1;
    mlp->weights[2] = weights2;

    free_variable(&(mlp->bias[0]));
    free_variable(&(mlp->bias[1]));
    free_variable(&(mlp->bias[2]));
    mlp->bias[0] = bias0;
    mlp->bias[1] = bias1;
    mlp->bias[2] = bias2;

    variable *output = forward_multilayer_perceptron(mlp, input);
    free_ndarray(&(output->grad));
    output->grad = up_stream_grad;

    backward_variable(output);

    assert_true(is_equal(output->val, output_val_gt));
    assert_true(is_equal(mlp->weights[0]->grad, weights0_grad_gt));
    assert_true(is_equal(mlp->weights[1]->grad, weights1_grad_gt));
    assert_true(is_equal(mlp->weights[2]->grad, weights2_grad_gt));
    assert_true(is_equal(mlp->bias[0]->grad, bias0_grad_gt));
    assert_true(is_equal(mlp->bias[1]->grad, bias1_grad_gt));
    assert_true(is_equal(mlp->bias[2]->grad, bias2_grad_gt));

    free_ndarray(&input_val);
    free_ndarray(&output_val_gt);
    free_ndarray(&weights0_val);
    free_ndarray(&weights1_val);
    free_ndarray(&weights2_val);
    free_ndarray(&bias0_val);
    free_ndarray(&bias1_val);
    free_ndarray(&bias2_val);
    free_ndarray(&weights0_grad_gt);
    free_ndarray(&weights1_grad_gt);
    free_ndarray(&weights2_grad_gt);
    free_ndarray(&bias0_grad_gt);
    free_ndarray(&bias1_grad_gt);
    free_ndarray(&bias2_grad_gt);

    free_variable(&output);

    free_multilayer_perceptron(&mlp);
}

int main(void)
{
    const struct CMUnitTest tests[] = {
        cmocka_unit_test(test_new_multilayer_perceptron),
        cmocka_unit_test(test_forward_multilayer_perceptron),

    };
    return cmocka_run_group_tests(tests, NULL, NULL);
}