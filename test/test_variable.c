#include <stdint.h>
#include <stddef.h>
#include <math.h>
#include <setjmp.h>
#include <stdarg.h>
#include "../extern/cmocka/include/cmocka.h"
#include "../src/variable.h"

const char *dataDir;

static void test_new_variable(void **state) {
  (void)state;

  ndarray *val = ones_ndarray(1, (int[]){4});
  variable *var = new_variable(val);

  assert_non_null(var);
  assert_int_equal(var->val->dim, 1);
  assert_int_equal(var->val->size, 4);
  assert_int_equal(var->val->shape[0], 4);
  for (int i = 0; i < var->val->size; i++) {
    assert_float_equal(var->val->data[i], 1.0, NDARRAY_TYPE_EPSILON);
  }
  assert_int_equal(var->grad->dim, 1);
  assert_int_equal(var->grad->size, 4);
  assert_int_equal(var->grad->shape[0], 4);
  for (int i = 0; i < var->grad->size; i++) {
    assert_float_equal(var->grad->data[i], 0.0, NDARRAY_TYPE_EPSILON);
  }
  assert_null(var->children);
  assert_int_equal(var->n_children, 0);
  assert_null(var->backward);
  assert_int_equal(var->ref_count, 0);

  free_ndarray(&val);
  free_graph_variable(&var);
}

static void test_shallow_copy_variable(void **state) {
  (void)state;
  ndarray *val = random_uniform_ndarray(2, (int[]){100, 100}, 0.0, 1.0);
  variable *var = new_variable(val);

  variable * n_var = shallow_copy_variable(var);
  assert_true(is_equal_ndarray(var->val, n_var->val, NDARRAY_TYPE_EPSILON));
  assert_true(is_equal_ndarray(var->grad, n_var->grad, NDARRAY_TYPE_EPSILON));
  assert_null(n_var->children);
  assert_int_equal(n_var->n_children, 0);
  assert_null(n_var->backward);
  assert_int_equal(n_var->ref_count, 0);

  free_ndarray(&val);
  free_graph_variable(&var);
  free_graph_variable(&n_var);
}

static void test_add_variable(void **state) {
  (void)state;

  char path[256];
  int num_tests = 6;

  for (int i = 0; i < num_tests; ++i) {
    snprintf(path, sizeof(path), "%s/test_add_variable/a%d.txt", dataDir, i);
    ndarray *a = read_ndarray(path);

    snprintf(path, sizeof(path), "%s/test_add_variable/b%d.txt", dataDir, i);
    ndarray *b = read_ndarray(path);

    snprintf(path, sizeof(path), "%s/test_add_variable/c%d.txt", dataDir, i);
    ndarray *c = read_ndarray(path);

    snprintf(path, sizeof(path), "%s/test_add_variable/d%d.txt", dataDir, i);
    ndarray *d = read_ndarray(path);

    snprintf(path, sizeof(path), "%s/test_add_variable/a%d_grad.txt", dataDir,
             i);
    ndarray *a_grad = read_ndarray(path);

    snprintf(path, sizeof(path), "%s/test_add_variable/b%d_grad.txt", dataDir,
             i);
    ndarray *b_grad = read_ndarray(path);

    variable *var_a = new_variable(a);
    variable *var_b = new_variable(b);
    variable *var_c_hat = add_variable(var_a, var_b);

    free_ndarray(&(var_c_hat->grad));
    var_c_hat->grad = d;
    backward_variable(var_c_hat);

    assert_true(is_equal_ndarray(var_c_hat->val, c, NDARRAY_TYPE_EPSILON));
    assert_true(is_equal_ndarray(var_a->grad, a_grad, NDARRAY_TYPE_EPSILON));
    assert_true(is_equal_ndarray(var_b->grad, b_grad, NDARRAY_TYPE_EPSILON));

    free_ndarray(&a);
    free_ndarray(&b);
    free_ndarray(&c);
    free_ndarray(&a_grad);
    free_ndarray(&b_grad);

    free_graph_variable(&var_c_hat);
  }
}

static void test_subtract_variable(void **state) {
  (void)state;

  char path[256];
  int num_tests = 6;

  for (int i = 0; i < num_tests; ++i) {
    snprintf(path, sizeof(path), "%s/test_subtract_variable/a%d.txt", dataDir,
             i);
    ndarray *a = read_ndarray(path);

    snprintf(path, sizeof(path), "%s/test_subtract_variable/b%d.txt", dataDir,
             i);
    ndarray *b = read_ndarray(path);

    snprintf(path, sizeof(path), "%s/test_subtract_variable/c%d.txt", dataDir,
             i);
    ndarray *c = read_ndarray(path);

    snprintf(path, sizeof(path), "%s/test_subtract_variable/d%d.txt", dataDir,
             i);
    ndarray *d = read_ndarray(path);

    snprintf(path, sizeof(path), "%s/test_subtract_variable/a%d_grad.txt",
             dataDir, i);
    ndarray *a_grad = read_ndarray(path);

    snprintf(path, sizeof(path), "%s/test_subtract_variable/b%d_grad.txt",
             dataDir, i);
    ndarray *b_grad = read_ndarray(path);

    variable *var_a = new_variable(a);
    variable *var_b = new_variable(b);
    variable *var_c_hat = subtract_variable(var_a, var_b);

    free_ndarray(&(var_c_hat->grad));
    var_c_hat->grad = d;
    backward_variable(var_c_hat);

    assert_true(is_equal_ndarray(var_c_hat->val, c, NDARRAY_TYPE_EPSILON));
    assert_true(is_equal_ndarray(var_a->grad, a_grad, NDARRAY_TYPE_EPSILON));
    assert_true(is_equal_ndarray(var_b->grad, b_grad, NDARRAY_TYPE_EPSILON));

    free_ndarray(&a);
    free_ndarray(&b);
    free_ndarray(&c);
    free_ndarray(&a_grad);
    free_ndarray(&b_grad);

    free_graph_variable(&var_c_hat);
  }
}

static void test_multiply_variable(void **state) {
  (void)state;

  char path[256];
  int num_tests = 6;

  for (int i = 0; i < num_tests; ++i) {
    snprintf(path, sizeof(path), "%s/test_multiply_variable/a%d.txt", dataDir,
             i);
    ndarray *a = read_ndarray(path);

    snprintf(path, sizeof(path), "%s/test_multiply_variable/b%d.txt", dataDir,
             i);
    ndarray *b = read_ndarray(path);

    snprintf(path, sizeof(path), "%s/test_multiply_variable/c%d.txt", dataDir,
             i);
    ndarray *c = read_ndarray(path);

    snprintf(path, sizeof(path), "%s/test_multiply_variable/d%d.txt", dataDir,
             i);
    ndarray *d = read_ndarray(path);

    snprintf(path, sizeof(path), "%s/test_multiply_variable/a%d_grad.txt",
             dataDir, i);
    ndarray *a_grad = read_ndarray(path);

    snprintf(path, sizeof(path), "%s/test_multiply_variable/b%d_grad.txt",
             dataDir, i);
    ndarray *b_grad = read_ndarray(path);

    variable *var_a = new_variable(a);
    variable *var_b = new_variable(b);
    variable *var_c_hat = multiply_variable(var_a, var_b);

    free_ndarray(&(var_c_hat->grad));
    var_c_hat->grad = d;
    backward_variable(var_c_hat);

    assert_true(is_equal_ndarray(var_c_hat->val, c, NDARRAY_TYPE_EPSILON));
    assert_true(is_equal_ndarray(var_a->grad, a_grad, NDARRAY_TYPE_EPSILON));
    assert_true(is_equal_ndarray(var_b->grad, b_grad, NDARRAY_TYPE_EPSILON));

    free_ndarray(&a);
    free_ndarray(&b);
    free_ndarray(&c);
    free_ndarray(&a_grad);
    free_ndarray(&b_grad);

    free_graph_variable(&var_c_hat);
  }
}

static void test_divide_variable(void **state) {
  (void)state;

  char path[256];
  int num_tests = 6;

  for (int i = 0; i < num_tests; ++i) {
    snprintf(path, sizeof(path), "%s/test_divide_variable/a%d.txt", dataDir, i);
    ndarray *a = read_ndarray(path);

    snprintf(path, sizeof(path), "%s/test_divide_variable/b%d.txt", dataDir, i);
    ndarray *b = read_ndarray(path);

    snprintf(path, sizeof(path), "%s/test_divide_variable/c%d.txt", dataDir, i);
    ndarray *c = read_ndarray(path);

    snprintf(path, sizeof(path), "%s/test_divide_variable/d%d.txt", dataDir, i);
    ndarray *d = read_ndarray(path);

    snprintf(path, sizeof(path), "%s/test_divide_variable/a%d_grad.txt",
             dataDir, i);
    ndarray *a_grad = read_ndarray(path);

    snprintf(path, sizeof(path), "%s/test_divide_variable/b%d_grad.txt",
             dataDir, i);
    ndarray *b_grad = read_ndarray(path);

    variable *var_a = new_variable(a);
    variable *var_b = new_variable(b);
    variable *var_c_hat = divide_variable(var_a, var_b);

    free_ndarray(&(var_c_hat->grad));
    var_c_hat->grad = d;
    backward_variable(var_c_hat);

    assert_true(is_equal_ndarray(var_c_hat->val, c, NDARRAY_TYPE_EPSILON));
    assert_true(is_equal_ndarray(var_a->grad, a_grad, NDARRAY_TYPE_EPSILON));
    assert_true(is_equal_ndarray(var_b->grad, b_grad, NDARRAY_TYPE_EPSILON));

    free_ndarray(&a);
    free_ndarray(&b);
    free_ndarray(&c);
    free_ndarray(&a_grad);
    free_ndarray(&b_grad);

    free_graph_variable(&var_c_hat);
  }
}

static void test_power_variable(void **state) {
  (void)state;

  char path[256];
  int num_tests = 6;

  for (int i = 0; i < num_tests; ++i) {
    snprintf(path, sizeof(path), "%s/test_power_variable/a%d.txt", dataDir, i);
    ndarray *a = read_ndarray(path);

    snprintf(path, sizeof(path), "%s/test_power_variable/b%d.txt", dataDir, i);
    ndarray *b = read_ndarray(path);

    snprintf(path, sizeof(path), "%s/test_power_variable/c%d.txt", dataDir, i);
    ndarray *c = read_ndarray(path);

    snprintf(path, sizeof(path), "%s/test_power_variable/d%d.txt", dataDir, i);
    ndarray *d = read_ndarray(path);

    snprintf(path, sizeof(path), "%s/test_power_variable/a%d_grad.txt", dataDir,
             i);
    ndarray *a_grad = read_ndarray(path);

    snprintf(path, sizeof(path), "%s/test_power_variable/b%d_grad.txt", dataDir,
             i);
    ndarray *b_grad = read_ndarray(path);

    variable *var_a = new_variable(a);
    variable *var_b = new_variable(b);
    variable *var_c_hat = power_variable(var_a, var_b);

    free_ndarray(&(var_c_hat->grad));
    var_c_hat->grad = d;
    backward_variable(var_c_hat);

    assert_true(is_equal_ndarray(var_c_hat->val, c, NDARRAY_TYPE_EPSILON));
    assert_true(is_equal_ndarray(var_a->grad, a_grad, NDARRAY_TYPE_EPSILON));
    assert_true(is_equal_ndarray(var_b->grad, b_grad, NDARRAY_TYPE_EPSILON));

    free_ndarray(&a);
    free_ndarray(&b);
    free_ndarray(&c);
    free_ndarray(&a_grad);
    free_ndarray(&b_grad);

    free_graph_variable(&var_c_hat);
  }
}

static void test_negate_variable(void **state) {
  (void)state;
  char path[256];

  snprintf(path, sizeof(path), "%s/test_negate_variable/x.txt", dataDir);
  ndarray *x = read_ndarray(path);

  snprintf(path, sizeof(path), "%s/test_negate_variable/y.txt", dataDir);
  ndarray *y = read_ndarray(path);

  snprintf(path, sizeof(path), "%s/test_negate_variable/z.txt", dataDir);
  ndarray *z = read_ndarray(path);

  snprintf(path, sizeof(path), "%s/test_negate_variable/x_grad.txt", dataDir);
  ndarray *x_grad = read_ndarray(path);

  variable *var_x = new_variable(x);
  variable *var_y_hat = negate_variable(var_x);
  free_ndarray(&(var_y_hat->grad));
  var_y_hat->grad = z;
  backward_variable(var_y_hat);

  assert_true(is_equal_ndarray(var_y_hat->val, y, NDARRAY_TYPE_EPSILON));
  assert_true(is_equal_ndarray(var_x->grad, x_grad, NDARRAY_TYPE_EPSILON));

  free_ndarray(&x);
  free_ndarray(&y);
  free_ndarray(&x_grad);

  free_graph_variable(&var_y_hat);
}

static void test_exp_variable(void **state) {
  (void)state;
  char path[256];

  snprintf(path, sizeof(path), "%s/test_exp_variable/x.txt", dataDir);
  ndarray *x = read_ndarray(path);

  snprintf(path, sizeof(path), "%s/test_exp_variable/y.txt", dataDir);
  ndarray *y = read_ndarray(path);

  snprintf(path, sizeof(path), "%s/test_exp_variable/z.txt", dataDir);
  ndarray *z = read_ndarray(path);

  snprintf(path, sizeof(path), "%s/test_exp_variable/x_grad.txt", dataDir);
  ndarray *x_grad = read_ndarray(path);

  variable *var_x = new_variable(x);
  variable *var_y_hat = exp_variable(var_x);
  free_ndarray(&(var_y_hat->grad));
  var_y_hat->grad = z;
  backward_variable(var_y_hat);

  assert_true(is_equal_ndarray(var_y_hat->val, y, NDARRAY_TYPE_EPSILON));
  assert_true(is_equal_ndarray(var_x->grad, x_grad, NDARRAY_TYPE_EPSILON));

  free_ndarray(&x);
  free_ndarray(&y);
  free_ndarray(&x_grad);

  free_graph_variable(&var_y_hat);
}

static void test_log_variable(void **state) {
  (void)state;
  char path[256];

  snprintf(path, sizeof(path), "%s/test_log_variable/x.txt", dataDir);
  ndarray *x = read_ndarray(path);

  snprintf(path, sizeof(path), "%s/test_log_variable/y.txt", dataDir);
  ndarray *y = read_ndarray(path);

  snprintf(path, sizeof(path), "%s/test_log_variable/z.txt", dataDir);
  ndarray *z = read_ndarray(path);

  snprintf(path, sizeof(path), "%s/test_log_variable/x_grad.txt", dataDir);
  ndarray *x_grad = read_ndarray(path);

  variable *var_x = new_variable(x);
  variable *var_y_hat = log_variable(var_x);
  free_ndarray(&(var_y_hat->grad));
  var_y_hat->grad = z;
  backward_variable(var_y_hat);

  assert_true(is_equal_ndarray(var_y_hat->val, y, NDARRAY_TYPE_EPSILON));
  assert_true(is_equal_ndarray(var_x->grad, x_grad, NDARRAY_TYPE_EPSILON));

  free_ndarray(&x);
  free_ndarray(&y);
  free_ndarray(&x_grad);

  free_graph_variable(&var_y_hat);
}

static void test_relu_variable(void **state) {
  (void)state;
  char path[256];

  snprintf(path, sizeof(path), "%s/test_relu_variable/x.txt", dataDir);
  ndarray *x = read_ndarray(path);

  snprintf(path, sizeof(path), "%s/test_relu_variable/y.txt", dataDir);
  ndarray *y = read_ndarray(path);

  snprintf(path, sizeof(path), "%s/test_relu_variable/z.txt", dataDir);
  ndarray *z = read_ndarray(path);

  snprintf(path, sizeof(path), "%s/test_relu_variable/x_grad.txt", dataDir);
  ndarray *x_grad = read_ndarray(path);

  variable *var_x = new_variable(x);
  variable *var_y_hat = relu_variable(var_x);
  free_ndarray(&(var_y_hat->grad));
  var_y_hat->grad = z;
  backward_variable(var_y_hat);

  assert_true(is_equal_ndarray(var_y_hat->val, y, NDARRAY_TYPE_EPSILON));
  assert_true(is_equal_ndarray(var_x->grad, x_grad, NDARRAY_TYPE_EPSILON));

  free_ndarray(&x);
  free_ndarray(&y);
  free_ndarray(&x_grad);

  free_graph_variable(&var_y_hat);
}

static void test_sigmoid_variable(void **state) {
  (void)state;
  char path[256];

  snprintf(path, sizeof(path), "%s/test_sigmoid_variable/x.txt", dataDir);
  ndarray *x = read_ndarray(path);

  snprintf(path, sizeof(path), "%s/test_sigmoid_variable/y.txt", dataDir);
  ndarray *y = read_ndarray(path);

  snprintf(path, sizeof(path), "%s/test_sigmoid_variable/z.txt", dataDir);
  ndarray *z = read_ndarray(path);

  snprintf(path, sizeof(path), "%s/test_sigmoid_variable/x_grad.txt", dataDir);
  ndarray *x_grad = read_ndarray(path);

  variable *var_x = new_variable(x);
  variable *var_y_hat = sigmoid_variable(var_x);
  free_ndarray(&(var_y_hat->grad));
  var_y_hat->grad = z;
  backward_variable(var_y_hat);

  assert_true(is_equal_ndarray(var_y_hat->val, y, NDARRAY_TYPE_EPSILON));
  assert_true(is_equal_ndarray(var_x->grad, x_grad, NDARRAY_TYPE_EPSILON));

  free_ndarray(&x);
  free_ndarray(&y);
  free_ndarray(&x_grad);

  free_graph_variable(&var_y_hat);
}

static void test_tanh_variable(void **state) {
  (void)state;
  char path[256];

  snprintf(path, sizeof(path), "%s/test_tanh_variable/x.txt", dataDir);
  ndarray *x = read_ndarray(path);

  snprintf(path, sizeof(path), "%s/test_tanh_variable/y.txt", dataDir);
  ndarray *y = read_ndarray(path);

  snprintf(path, sizeof(path), "%s/test_tanh_variable/z.txt", dataDir);
  ndarray *z = read_ndarray(path);

  snprintf(path, sizeof(path), "%s/test_tanh_variable/x_grad.txt", dataDir);
  ndarray *x_grad = read_ndarray(path);

  variable *var_x = new_variable(x);
  variable *var_y_hat = tanh_variable(var_x);
  free_ndarray(&(var_y_hat->grad));
  var_y_hat->grad = z;
  backward_variable(var_y_hat);

  assert_true(is_equal_ndarray(var_y_hat->val, y, NDARRAY_TYPE_EPSILON));
  assert_true(is_equal_ndarray(var_x->grad, x_grad, NDARRAY_TYPE_EPSILON));

  free_ndarray(&x);
  free_ndarray(&y);
  free_ndarray(&x_grad);

  free_graph_variable(&var_y_hat);
}

static void test_sum_variable(void **state) {
  (void)state;
  char path[256];

  int axes[3] = {0, 1, 2};
  ndarray *y_hat[3];

  for (int i = 0; i < 3; ++i) {
    snprintf(path, sizeof(path), "%s/test_sum_variable/x%d.txt", dataDir, i);
    ndarray *x = read_ndarray(path);

    snprintf(path, sizeof(path), "%s/test_sum_variable/y%d.txt", dataDir, i);
    ndarray *y = read_ndarray(path);

    snprintf(path, sizeof(path), "%s/test_sum_variable/z%d.txt", dataDir, i);
    ndarray *z = read_ndarray(path);

    snprintf(path, sizeof(path), "%s/test_sum_variable/x%d_grad.txt", dataDir,
             i);
    ndarray *x_grad = read_ndarray(path);

    variable *var_x = new_variable(x);
    variable *var_y_hat = sum_variable(var_x, axes[i]);
    free_ndarray(&(var_y_hat->grad));
    var_y_hat->grad = z;
    backward_variable(var_y_hat);

    assert_true(is_equal_ndarray(var_y_hat->val, y, NDARRAY_TYPE_EPSILON));
    assert_true(is_equal_ndarray(var_x->grad, x_grad, NDARRAY_TYPE_EPSILON));

    free_ndarray(&x);
    free_ndarray(&y);
    free_ndarray(&x_grad);

    free_graph_variable(&var_y_hat);
  }
}

static void test_softmax_variable(void **state) {
  (void)state;
  char path[256];

  int axes[3] = {0, 1, 2};
  ndarray *y_hat[3];

  for (int i = 0; i < 3; ++i) {
    snprintf(path, sizeof(path), "%s/test_softmax_variable/x%d.txt", dataDir,
             i);
    ndarray *x = read_ndarray(path);

    snprintf(path, sizeof(path), "%s/test_softmax_variable/y%d.txt", dataDir,
             i);
    ndarray *y = read_ndarray(path);

    snprintf(path, sizeof(path), "%s/test_softmax_variable/z%d.txt", dataDir,
             i);
    ndarray *z = read_ndarray(path);

    snprintf(path, sizeof(path), "%s/test_softmax_variable/x%d_grad.txt",
             dataDir, i);
    ndarray *x_grad = read_ndarray(path);

    variable *var_x = new_variable(x);
    variable *var_y_hat = softmax_variable(var_x, axes[i]);
    free_ndarray(&(var_y_hat->grad));
    var_y_hat->grad = z;
    backward_variable(var_y_hat);

    assert_true(is_equal_ndarray(var_y_hat->val, y, NDARRAY_TYPE_EPSILON));
    assert_true(is_equal_ndarray(var_x->grad, x_grad, NDARRAY_TYPE_EPSILON));

    free_ndarray(&x);
    free_ndarray(&y);
    free_ndarray(&x_grad);

    free_graph_variable(&var_y_hat);
  }
}

static void test_matmul_variable(void **state) {
  (void)state;

  char path[256];
  int num_tests = 4;

  for (int i = 0; i < num_tests; ++i) {
    snprintf(path, sizeof(path), "%s/test_matmul_variable/a%d.txt", dataDir, i);
    ndarray *a = read_ndarray(path);

    snprintf(path, sizeof(path), "%s/test_matmul_variable/b%d.txt", dataDir, i);
    ndarray *b = read_ndarray(path);

    snprintf(path, sizeof(path), "%s/test_matmul_variable/c%d.txt", dataDir, i);
    ndarray *c = read_ndarray(path);

    snprintf(path, sizeof(path), "%s/test_matmul_variable/d%d.txt", dataDir, i);
    ndarray *d = read_ndarray(path);

    snprintf(path, sizeof(path), "%s/test_matmul_variable/a%d_grad.txt",
             dataDir, i);
    ndarray *a_grad = read_ndarray(path);

    snprintf(path, sizeof(path), "%s/test_matmul_variable/b%d_grad.txt",
             dataDir, i);
    ndarray *b_grad = read_ndarray(path);

    variable *var_a = new_variable(a);
    variable *var_b = new_variable(b);
    variable *var_c_hat = matmul_variable(var_a, var_b);

    free_ndarray(&(var_c_hat->grad));
    var_c_hat->grad = d;
    backward_variable(var_c_hat);

    assert_true(is_equal_ndarray(var_c_hat->val, c, NDARRAY_TYPE_EPSILON));
    assert_true(is_equal_ndarray(var_a->grad, a_grad, NDARRAY_TYPE_EPSILON));
    assert_true(is_equal_ndarray(var_b->grad, b_grad, NDARRAY_TYPE_EPSILON));

    free_ndarray(&a);
    free_ndarray(&b);
    free_ndarray(&c);
    free_ndarray(&a_grad);
    free_ndarray(&b_grad);

    free_graph_variable(&var_c_hat);
  }
}

static void test_backward_variable(void **state) {
  (void)state;
  char path[256];

  snprintf(path, sizeof(path), "%s/test_backward_variable/a.txt", dataDir);
  ndarray *a = read_ndarray(path);

  snprintf(path, sizeof(path), "%s/test_backward_variable/b.txt", dataDir);
  ndarray *b = read_ndarray(path);

  snprintf(path, sizeof(path), "%s/test_backward_variable/c.txt", dataDir);
  ndarray *c = read_ndarray(path);

  snprintf(path, sizeof(path), "%s/test_backward_variable/d.txt", dataDir);
  ndarray *d = read_ndarray(path);

  snprintf(path, sizeof(path), "%s/test_backward_variable/e.txt", dataDir);
  ndarray *e = read_ndarray(path);

  snprintf(path, sizeof(path), "%s/test_backward_variable/a_grad.txt", dataDir);
  ndarray *a_grad = read_ndarray(path);

  snprintf(path, sizeof(path), "%s/test_backward_variable/b_grad.txt", dataDir);
  ndarray *b_grad = read_ndarray(path);

  snprintf(path, sizeof(path), "%s/test_backward_variable/c_grad.txt", dataDir);
  ndarray *c_grad = read_ndarray(path);

  variable *var_a = new_variable(a);
  variable *var_b = new_variable(b);
  variable *var_c = new_variable(c);
  variable *var_d_hat = add_variable(
      multiply_variable(var_a,
                        add_variable(multiply_variable(var_a, var_b), var_c)),
      var_c);
  free_ndarray(&(var_d_hat->grad));
  var_d_hat->grad = e;
  backward_variable(var_d_hat);

  assert_true(is_equal_ndarray(var_d_hat->val, d, NDARRAY_TYPE_EPSILON));
  assert_true(is_equal_ndarray(var_a->grad, a_grad, NDARRAY_TYPE_EPSILON));
  assert_true(is_equal_ndarray(var_b->grad, b_grad, NDARRAY_TYPE_EPSILON));
  assert_true(is_equal_ndarray(var_c->grad, c_grad, NDARRAY_TYPE_EPSILON));

  free_ndarray(&a);
  free_ndarray(&b);
  free_ndarray(&c);
  free_ndarray(&d);
  free_ndarray(&a_grad);
  free_ndarray(&b_grad);
  free_ndarray(&c_grad);
  free_graph_variable(&var_d_hat);
}

int main(void) {
  dataDir = getenv("TEST_VARIABLE_DATA_DIR");
  if (dataDir == NULL) {
    fprintf(stderr,
            "Error: TEST_VARIABLE_DATA_DIR environment variable not set.\n");
    return 1;
  }
  const struct CMUnitTest tests[] = {
      cmocka_unit_test(test_new_variable),
      cmocka_unit_test(test_shallow_copy_variable),
      cmocka_unit_test(test_add_variable),
      cmocka_unit_test(test_subtract_variable),
      cmocka_unit_test(test_multiply_variable),
      cmocka_unit_test(test_divide_variable),
      cmocka_unit_test(test_power_variable),
      cmocka_unit_test(test_negate_variable),
      cmocka_unit_test(test_exp_variable),
      cmocka_unit_test(test_log_variable),
      cmocka_unit_test(test_relu_variable),
      cmocka_unit_test(test_sigmoid_variable),
      cmocka_unit_test(test_tanh_variable),
      cmocka_unit_test(test_sum_variable),
      cmocka_unit_test(test_softmax_variable),
      cmocka_unit_test(test_matmul_variable),
      cmocka_unit_test(test_backward_variable),
  };
  return cmocka_run_group_tests(tests, NULL, NULL);
}
