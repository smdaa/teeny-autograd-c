#include <math.h>
#include <setjmp.h>
#include <stdarg.h>
#include <stddef.h>
#include <stdlib.h>
#include <cmocka.h>

#include "../src/multilayer_perceptron.h"

static void test_new_multilayer_perceptron(void **state) {
  (void)state;

  multilayer_perceptron *mlp =
      new_multilayer_perceptron(2, 64, (int[]){16, 32}, (int[]){32, 8},
                                (activation_function[]){LINEAR, LINEAR},
                                (random_initialisation[]){UNIFORM, UNIFORM});
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

  free_graph_variable(&(mlp->weights[0]));
  free_graph_variable(&(mlp->weights[1]));
  free_graph_variable(&(mlp->bias[0]));
  free_graph_variable(&(mlp->bias[1]));
  free_multilayer_perceptron(&mlp);
}

static void test_forward_multilayer_perceptron(void **state) {
  (void)state;

  ndarray *x = read_ndarray("./test_data/test_multilayer_perceptron/"
                            "test_forward_multilayer_perceptron/x.txt");
  ndarray *y = read_ndarray("./test_data/test_multilayer_perceptron/"
                            "test_forward_multilayer_perceptron/y.txt");
  ndarray *weights0 =
      read_ndarray("./test_data/test_multilayer_perceptron/"
                   "test_forward_multilayer_perceptron/weights0.txt");
  ndarray *weights1 =
      read_ndarray("./test_data/test_multilayer_perceptron/"
                   "test_forward_multilayer_perceptron/weights1.txt");
  ndarray *weights2 =
      read_ndarray("./test_data/test_multilayer_perceptron/"
                   "test_forward_multilayer_perceptron/weights2.txt");
  ndarray *bias0 = read_ndarray("./test_data/test_multilayer_perceptron/"
                                "test_forward_multilayer_perceptron/bias0.txt");
  ndarray *bias1 = read_ndarray("./test_data/test_multilayer_perceptron/"
                                "test_forward_multilayer_perceptron/bias1.txt");
  ndarray *bias2 = read_ndarray("./test_data/test_multilayer_perceptron/"
                                "test_forward_multilayer_perceptron/bias2.txt");
  ndarray *z = read_ndarray("./test_data/test_multilayer_perceptron/"
                            "test_forward_multilayer_perceptron/z.txt");
  ndarray *weights0_grad =
      read_ndarray("./test_data/test_multilayer_perceptron/"
                   "test_forward_multilayer_perceptron/weights0_grad.txt");
  ndarray *weights1_grad =
      read_ndarray("./test_data/test_multilayer_perceptron/"
                   "test_forward_multilayer_perceptron/weights1_grad.txt");
  ndarray *weights2_grad =
      read_ndarray("./test_data/test_multilayer_perceptron/"
                   "test_forward_multilayer_perceptron/weights2_grad.txt");
  ndarray *bias0_grad =
      read_ndarray("./test_data/test_multilayer_perceptron/"
                   "test_forward_multilayer_perceptron/bias0_grad.txt");
  ndarray *bias1_grad =
      read_ndarray("./test_data/test_multilayer_perceptron/"
                   "test_forward_multilayer_perceptron/bias1_grad.txt");
  ndarray *bias2_grad =
      read_ndarray("./test_data/test_multilayer_perceptron/"
                   "test_forward_multilayer_perceptron/bias2_grad.txt");

  variable *x_var = new_variable(x);
  variable *weights0_var = new_variable(weights0);
  variable *weights1_var = new_variable(weights1);
  variable *weights2_var = new_variable(weights2);
  variable *bias0_var = new_variable(bias0);
  variable *bias1_var = new_variable(bias1);
  variable *bias2_var = new_variable(bias2);

  multilayer_perceptron *mlp = new_multilayer_perceptron(
      3, 64, (int[]){32, 128, 256}, (int[]){128, 256, 512},
      (activation_function[]){LINEAR, LINEAR, LINEAR},
      (random_initialisation[]){UNIFORM, UNIFORM, UNIFORM});
  free_graph_variable(&(mlp->weights[0]));
  free_graph_variable(&(mlp->weights[1]));
  free_graph_variable(&(mlp->weights[2]));
  mlp->weights[0] = weights0_var;
  mlp->weights[1] = weights1_var;
  mlp->weights[2] = weights2_var;
  free_graph_variable(&(mlp->bias[0]));
  free_graph_variable(&(mlp->bias[1]));
  free_graph_variable(&(mlp->bias[2]));
  mlp->bias[0] = bias0_var;
  mlp->bias[1] = bias1_var;
  mlp->bias[2] = bias2_var;
  variable *y_hat_var = forward_multilayer_perceptron(mlp, x_var);
  free_ndarray(&(y_hat_var->grad));
  y_hat_var->grad = z;
  backward_variable(y_hat_var);

  assert_true(is_equal_ndarray(y_hat_var->val, y, 2 * NDARRAY_TYPE_EPSILON));
  assert_true(is_equal_ndarray(mlp->weights[0]->grad, weights0_grad,
                               NDARRAY_TYPE_EPSILON));
  assert_true(is_equal_ndarray(mlp->weights[1]->grad, weights1_grad,
                               NDARRAY_TYPE_EPSILON));
  assert_true(is_equal_ndarray(mlp->weights[2]->grad, weights2_grad,
                               NDARRAY_TYPE_EPSILON));
  assert_true(
      is_equal_ndarray(mlp->bias[0]->grad, bias0_grad, NDARRAY_TYPE_EPSILON));
  assert_true(
      is_equal_ndarray(mlp->bias[1]->grad, bias1_grad, NDARRAY_TYPE_EPSILON));
  assert_true(
      is_equal_ndarray(mlp->bias[2]->grad, bias2_grad, NDARRAY_TYPE_EPSILON));

  free_ndarray(&x);
  free_ndarray(&y);
  free_ndarray(&weights0);
  free_ndarray(&weights1);
  free_ndarray(&weights2);
  free_ndarray(&bias0);
  free_ndarray(&bias1);
  free_ndarray(&bias2);
  free_ndarray(&weights0_grad);
  free_ndarray(&weights1_grad);
  free_ndarray(&weights2_grad);
  free_ndarray(&bias0_grad);
  free_ndarray(&bias1_grad);
  free_ndarray(&bias2_grad);
  free_graph_variable(&y_hat_var);
  free_multilayer_perceptron(&mlp);
}
int main(void) {
  const struct CMUnitTest tests[] = {
      cmocka_unit_test(test_new_multilayer_perceptron),
      cmocka_unit_test(test_forward_multilayer_perceptron),

  };
  return cmocka_run_group_tests(tests, NULL, NULL);
}