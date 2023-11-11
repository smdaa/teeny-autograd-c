#include <stdlib.h>
#include <stdarg.h>
#include <stddef.h>
#include <setjmp.h>
#include <cmocka.h>
#include "../src/variable.h"

#define EPSILON 0.0001

static void test_new_variable(void **state)
{
    (void)state;

    ndarray *val = ones_ndarray(1, (int[]){4});
    variable *var = new_variable(val);

    assert_non_null(var);
    assert_int_equal(var->val->dim, 1);
    assert_int_equal(var->val->size, 4);
    assert_int_equal(var->val->shape[0], 4);
    for (int i = 0; i < var->val->size; i++)
    {
        assert_float_equal(var->val->data[i], 1.0, EPSILON);
    }
    assert_int_equal(var->grad->dim, 1);
    assert_int_equal(var->grad->size, 4);
    assert_int_equal(var->grad->shape[0], 4);
    for (int i = 0; i < var->grad->size; i++)
    {
        assert_float_equal(var->grad->data[i], 0.0, EPSILON);
    }
    assert_null(var->children);
    assert_int_equal(var->n_children, 0);
    assert_null(var->backward);
    assert_int_equal(var->ref_count, 0);

    free_ndarray(&val);
    free_variable(&var);
}

static void test_add_variable(void **state)
{
    (void)state;

    ndarray *val1 = read_ndarray("./test_data/test_add_variable_var1_val.txt");
    ndarray *val2 = read_ndarray("./test_data/test_add_variable_var2_val.txt");
    ndarray *var_val_gt = read_ndarray("./test_data/test_add_variable_var_val.txt");
    ndarray *up_stream_grad = read_ndarray("./test_data/test_add_variable_up_stream_grad.txt");
    ndarray *var1_grad_gt = read_ndarray("./test_data/test_add_variable_var1_grad.txt");
    ndarray *var2_grad_gt = read_ndarray("./test_data/test_add_variable_var2_grad.txt");

    variable *var2 = new_variable(val2);
    variable *var1 = new_variable(val1);
    variable *var = add_variable(var1, var2);

    free_ndarray(&(var->grad));
    var->grad = up_stream_grad;
    var->backward(var);

    assert_true(is_equal(var->val, var_val_gt));
    assert_true(is_equal(var1->grad, var1_grad_gt));
    assert_true(is_equal(var2->grad, var2_grad_gt));

    free_ndarray(&val1);
    free_ndarray(&val2);
    free_ndarray(&var_val_gt);
    free_ndarray(&var1_grad_gt);
    free_ndarray(&var2_grad_gt);

    free_variable(&var);
}

static void test_subtract_variable(void **state)
{
    (void)state;

    ndarray *val1 = read_ndarray("./test_data/test_subtract_variable_var1_val.txt");
    ndarray *val2 = read_ndarray("./test_data/test_subtract_variable_var2_val.txt");
    ndarray *var_val_gt = read_ndarray("./test_data/test_subtract_variable_var_val.txt");
    ndarray *up_stream_grad = read_ndarray("./test_data/test_subtract_variable_up_stream_grad.txt");
    ndarray *var1_grad_gt = read_ndarray("./test_data/test_subtract_variable_var1_grad.txt");
    ndarray *var2_grad_gt = read_ndarray("./test_data/test_subtract_variable_var2_grad.txt");

    variable *var2 = new_variable(val2);
    variable *var1 = new_variable(val1);
    variable *var = subtract_variable(var1, var2);

    free_ndarray(&(var->grad));
    var->grad = up_stream_grad;
    var->backward(var);

    assert_true(is_equal(var->val, var_val_gt));
    assert_true(is_equal(var1->grad, var1_grad_gt));
    assert_true(is_equal(var2->grad, var2_grad_gt));

    free_ndarray(&val1);
    free_ndarray(&val2);
    free_ndarray(&var_val_gt);
    free_ndarray(&var1_grad_gt);
    free_ndarray(&var2_grad_gt);

    free_variable(&var);
}

static void test_multiply_variable(void **state)
{
    (void)state;

    ndarray *val1 = read_ndarray("./test_data/test_multiply_variable_var1_val.txt");
    ndarray *val2 = read_ndarray("./test_data/test_multiply_variable_var2_val.txt");
    ndarray *var_val_gt = read_ndarray("./test_data/test_multiply_variable_var_val.txt");
    ndarray *up_stream_grad = read_ndarray("./test_data/test_multiply_variable_up_stream_grad.txt");
    ndarray *var1_grad_gt = read_ndarray("./test_data/test_multiply_variable_var1_grad.txt");
    ndarray *var2_grad_gt = read_ndarray("./test_data/test_multiply_variable_var2_grad.txt");

    variable *var2 = new_variable(val2);
    variable *var1 = new_variable(val1);
    variable *var = multiply_variable(var1, var2);

    free_ndarray(&(var->grad));
    var->grad = up_stream_grad;
    var->backward(var);

    assert_true(is_equal(var->val, var_val_gt));
    assert_true(is_equal(var1->grad, var1_grad_gt));
    assert_true(is_equal(var2->grad, var2_grad_gt));

    free_ndarray(&val1);
    free_ndarray(&val2);
    free_ndarray(&var_val_gt);
    free_ndarray(&var1_grad_gt);
    free_ndarray(&var2_grad_gt);

    free_variable(&var);
}

static void test_divide_variable(void **state)
{
    (void)state;

    ndarray *val1 = read_ndarray("./test_data/test_divide_variable_var1_val.txt");
    ndarray *val2 = read_ndarray("./test_data/test_divide_variable_var2_val.txt");
    ndarray *var_val_gt = read_ndarray("./test_data/test_divide_variable_var_val.txt");
    ndarray *up_stream_grad = read_ndarray("./test_data/test_divide_variable_up_stream_grad.txt");
    ndarray *var1_grad_gt = read_ndarray("./test_data/test_divide_variable_var1_grad.txt");
    ndarray *var2_grad_gt = read_ndarray("./test_data/test_divide_variable_var2_grad.txt");

    variable *var2 = new_variable(val2);
    variable *var1 = new_variable(val1);
    variable *var = divide_variable(var1, var2);

    free_ndarray(&(var->grad));
    var->grad = up_stream_grad;
    var->backward(var);

    assert_true(is_equal(var->val, var_val_gt));
    assert_true(is_equal(var1->grad, var1_grad_gt));
    assert_true(is_equal(var2->grad, var2_grad_gt));

    free_ndarray(&val1);
    free_ndarray(&val2);
    free_ndarray(&var_val_gt);
    free_ndarray(&var1_grad_gt);
    free_ndarray(&var2_grad_gt);

    free_variable(&var);
}

static void test_power_variable(void **state)
{
    (void)state;

    ndarray *val1 = read_ndarray("./test_data/test_power_variable_var1_val.txt");
    ndarray *val2 = read_ndarray("./test_data/test_power_variable_var2_val.txt");
    ndarray *var_val_gt = read_ndarray("./test_data/test_power_variable_var_val.txt");
    ndarray *up_stream_grad = read_ndarray("./test_data/test_power_variable_up_stream_grad.txt");
    ndarray *var1_grad_gt = read_ndarray("./test_data/test_power_variable_var1_grad.txt");
    ndarray *var2_grad_gt = read_ndarray("./test_data/test_power_variable_var2_grad.txt");

    variable *var2 = new_variable(val2);
    variable *var1 = new_variable(val1);
    variable *var = power_variable(var1, var2);

    free_ndarray(&(var->grad));
    var->grad = up_stream_grad;
    var->backward(var);

    assert_true(is_equal(var->val, var_val_gt));
    assert_true(is_equal(var1->grad, var1_grad_gt));
    assert_true(is_equal(var2->grad, var2_grad_gt));

    free_ndarray(&val1);
    free_ndarray(&val2);
    free_ndarray(&var_val_gt);
    free_ndarray(&var1_grad_gt);
    free_ndarray(&var2_grad_gt);

    free_variable(&var);
}

static void test_relu_variable(void **state)
{
    (void)state;

    ndarray *val = read_ndarray("./test_data/test_relu_variable_var_val.txt");
    ndarray *nvar_val_gt = read_ndarray("./test_data/test_relu_variable_nvar_val.txt");
    ndarray *up_stream_grad = read_ndarray("./test_data/test_relu_variable_up_stream_grad.txt");
    ndarray *var_grad_gt = read_ndarray("./test_data/test_relu_variable_var_grad.txt");

    variable *var = new_variable(val);
    variable *n_var = relu_variable(var);

    free_ndarray(&(n_var->grad));
    n_var->grad = up_stream_grad;
    n_var->backward(n_var);

    assert_true(is_equal(n_var->val, nvar_val_gt));
    assert_true(is_equal(var->grad, var_grad_gt));

    free_ndarray(&val);
    free_ndarray(&nvar_val_gt);
    free_ndarray(&var_grad_gt);

    free_variable(&n_var);
}

static void test_matmul_variable(void **state)
{
    (void)state;

    ndarray *val1 = read_ndarray("./test_data/test_matmul_variable_var1_val.txt");
    ndarray *val2 = read_ndarray("./test_data/test_matmul_variable_var2_val.txt");
    ndarray *var_val_gt = read_ndarray("./test_data/test_matmul_variable_var_val.txt");
    ndarray *up_stream_grad = read_ndarray("./test_data/test_matmul_variable_up_stream_grad.txt");
    ndarray *var1_grad_gt = read_ndarray("./test_data/test_matmul_variable_var1_grad.txt");
    ndarray *var2_grad_gt = read_ndarray("./test_data/test_matmul_variable_var2_grad.txt");

    variable *var2 = new_variable(val2);
    variable *var1 = new_variable(val1);
    variable *var = matmul_variable(var1, var2);

    free_ndarray(&(var->grad));
    var->grad = up_stream_grad;
    var->backward(var);

    assert_true(is_equal(var->val, var_val_gt));
    assert_true(is_equal(var1->grad, var1_grad_gt));
    assert_true(is_equal(var2->grad, var2_grad_gt));

    free_ndarray(&val1);
    free_ndarray(&val2);
    free_ndarray(&var_val_gt);
    free_ndarray(&var1_grad_gt);
    free_ndarray(&var2_grad_gt);

    free_variable(&var);
}

static void test_backward_variable(void **state)
{
    (void)state;

    ndarray *a_val = read_ndarray("./test_data/test_backward_a_val.txt");
    ndarray *b_val = read_ndarray("./test_data/test_backward_b_val.txt");
    ndarray *c_val_gt = read_ndarray("./test_data/test_backward_c_val.txt");
    ndarray *up_stream_grad = read_ndarray("./test_data/test_backward_up_stream_grad.txt");
    ndarray *a_grad_gt = read_ndarray("./test_data/test_backward_a_grad.txt");
    ndarray *b_grad_gt = read_ndarray("./test_data/test_backward_b_grad.txt");

    variable *a = new_variable(a_val);
    variable *b = new_variable(b_val);
    variable *c = add_variable(b, multiply_variable(a, add_variable(a, b)));

    free_ndarray(&(c->grad));
    c->grad = up_stream_grad;
    backward_variable(c);

    assert_true(is_equal(c->val, c_val_gt));
    assert_true(is_equal(a->grad, a_grad_gt));
    assert_true(is_equal(b->grad, b_grad_gt));

    free_ndarray(&a_val);
    free_ndarray(&b_val);
    free_ndarray(&a_grad_gt);
    free_ndarray(&b_grad_gt);
    free_ndarray(&c_val_gt);

    free_variable(&c);
}

int main(void)
{
    const struct CMUnitTest tests[] = {
        cmocka_unit_test(test_new_variable),
        cmocka_unit_test(test_add_variable),
        cmocka_unit_test(test_subtract_variable),
        cmocka_unit_test(test_multiply_variable),
        cmocka_unit_test(test_divide_variable),
        cmocka_unit_test(test_power_variable),
        cmocka_unit_test(test_relu_variable),
        cmocka_unit_test(test_matmul_variable),
        cmocka_unit_test(test_backward_variable),
    };
    return cmocka_run_group_tests(tests, NULL, NULL);
}
