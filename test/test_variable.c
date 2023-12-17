#include <setjmp.h>
#include <stdarg.h>
#include <stddef.h>
#include <stdlib.h>
#include <cmocka.h>

#include "../src/variable.h"

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

static void test_add_variable(void **state) {
  (void)state;
  ndarray *a0 =
      read_ndarray("./test_data/test_variable/test_add_variable/a0.txt");
  ndarray *a1 =
      read_ndarray("./test_data/test_variable/test_add_variable/a1.txt");
  ndarray *a2 =
      read_ndarray("./test_data/test_variable/test_add_variable/a2.txt");
  ndarray *a3 =
      read_ndarray("./test_data/test_variable/test_add_variable/a3.txt");
  ndarray *a4 =
      read_ndarray("./test_data/test_variable/test_add_variable/a4.txt");
  ndarray *a5 =
      read_ndarray("./test_data/test_variable/test_add_variable/a5.txt");
  ndarray *b0 =
      read_ndarray("./test_data/test_variable/test_add_variable/b0.txt");
  ndarray *b1 =
      read_ndarray("./test_data/test_variable/test_add_variable/b1.txt");
  ndarray *b2 =
      read_ndarray("./test_data/test_variable/test_add_variable/b2.txt");
  ndarray *b3 =
      read_ndarray("./test_data/test_variable/test_add_variable/b3.txt");
  ndarray *b4 =
      read_ndarray("./test_data/test_variable/test_add_variable/b4.txt");
  ndarray *b5 =
      read_ndarray("./test_data/test_variable/test_add_variable/b5.txt");
  ndarray *c0 =
      read_ndarray("./test_data/test_variable/test_add_variable/c0.txt");
  ndarray *c1 =
      read_ndarray("./test_data/test_variable/test_add_variable/c1.txt");
  ndarray *c2 =
      read_ndarray("./test_data/test_variable/test_add_variable/c2.txt");
  ndarray *c3 =
      read_ndarray("./test_data/test_variable/test_add_variable/c3.txt");
  ndarray *c4 =
      read_ndarray("./test_data/test_variable/test_add_variable/c4.txt");
  ndarray *c5 =
      read_ndarray("./test_data/test_variable/test_add_variable/c5.txt");
  ndarray *d0 =
      read_ndarray("./test_data/test_variable/test_add_variable/d0.txt");
  ndarray *d1 =
      read_ndarray("./test_data/test_variable/test_add_variable/d1.txt");
  ndarray *d2 =
      read_ndarray("./test_data/test_variable/test_add_variable/d2.txt");
  ndarray *d3 =
      read_ndarray("./test_data/test_variable/test_add_variable/d3.txt");
  ndarray *d4 =
      read_ndarray("./test_data/test_variable/test_add_variable/d4.txt");
  ndarray *d5 =
      read_ndarray("./test_data/test_variable/test_add_variable/d5.txt");
  ndarray *a0_grad =
      read_ndarray("./test_data/test_variable/test_add_variable/a0_grad.txt");
  ndarray *a1_grad =
      read_ndarray("./test_data/test_variable/test_add_variable/a1_grad.txt");
  ndarray *a2_grad =
      read_ndarray("./test_data/test_variable/test_add_variable/a2_grad.txt");
  ndarray *a3_grad =
      read_ndarray("./test_data/test_variable/test_add_variable/a3_grad.txt");
  ndarray *a4_grad =
      read_ndarray("./test_data/test_variable/test_add_variable/a4_grad.txt");
  ndarray *a5_grad =
      read_ndarray("./test_data/test_variable/test_add_variable/a5_grad.txt");
  ndarray *b0_grad =
      read_ndarray("./test_data/test_variable/test_add_variable/b0_grad.txt");
  ndarray *b1_grad =
      read_ndarray("./test_data/test_variable/test_add_variable/b1_grad.txt");
  ndarray *b2_grad =
      read_ndarray("./test_data/test_variable/test_add_variable/b2_grad.txt");
  ndarray *b3_grad =
      read_ndarray("./test_data/test_variable/test_add_variable/b3_grad.txt");
  ndarray *b4_grad =
      read_ndarray("./test_data/test_variable/test_add_variable/b4_grad.txt");
  ndarray *b5_grad =
      read_ndarray("./test_data/test_variable/test_add_variable/b5_grad.txt");

  variable *var_a0 = new_variable(a0);
  variable *var_a1 = new_variable(a1);
  variable *var_a2 = new_variable(a2);
  variable *var_a3 = new_variable(a3);
  variable *var_a4 = new_variable(a4);
  variable *var_a5 = new_variable(a5);
  variable *var_b0 = new_variable(b0);
  variable *var_b1 = new_variable(b1);
  variable *var_b2 = new_variable(b2);
  variable *var_b3 = new_variable(b3);
  variable *var_b4 = new_variable(b4);
  variable *var_b5 = new_variable(b5);
  variable *var_c0_hat = add_variable(var_a0, var_b0);
  variable *var_c1_hat = add_variable(var_a1, var_b1);
  variable *var_c2_hat = add_variable(var_a2, var_b2);
  variable *var_c3_hat = add_variable(var_a3, var_b3);
  variable *var_c4_hat = add_variable(var_a4, var_b4);
  variable *var_c5_hat = add_variable(var_a5, var_b5);
  free_ndarray(&(var_c0_hat->grad));
  free_ndarray(&(var_c1_hat->grad));
  free_ndarray(&(var_c2_hat->grad));
  free_ndarray(&(var_c3_hat->grad));
  free_ndarray(&(var_c4_hat->grad));
  free_ndarray(&(var_c5_hat->grad));
  var_c0_hat->grad = d0;
  var_c1_hat->grad = d1;
  var_c2_hat->grad = d2;
  var_c3_hat->grad = d3;
  var_c4_hat->grad = d4;
  var_c5_hat->grad = d5;
  backward_variable(var_c0_hat);
  backward_variable(var_c1_hat);
  backward_variable(var_c2_hat);
  backward_variable(var_c3_hat);
  backward_variable(var_c4_hat);
  backward_variable(var_c5_hat);

  assert_true(is_equal_ndarray(var_c0_hat->val, c0, NDARRAY_TYPE_EPSILON));
  assert_true(is_equal_ndarray(var_c1_hat->val, c1, NDARRAY_TYPE_EPSILON));
  assert_true(is_equal_ndarray(var_c2_hat->val, c2, NDARRAY_TYPE_EPSILON));
  assert_true(is_equal_ndarray(var_c3_hat->val, c3, NDARRAY_TYPE_EPSILON));
  assert_true(is_equal_ndarray(var_c4_hat->val, c4, NDARRAY_TYPE_EPSILON));
  assert_true(is_equal_ndarray(var_c5_hat->val, c5, NDARRAY_TYPE_EPSILON));
  assert_true(is_equal_ndarray(var_a0->grad, a0_grad, NDARRAY_TYPE_EPSILON));
  assert_true(is_equal_ndarray(var_a1->grad, a1_grad, NDARRAY_TYPE_EPSILON));
  assert_true(is_equal_ndarray(var_a2->grad, a2_grad, NDARRAY_TYPE_EPSILON));
  assert_true(is_equal_ndarray(var_a3->grad, a3_grad, NDARRAY_TYPE_EPSILON));
  assert_true(is_equal_ndarray(var_a4->grad, a4_grad, NDARRAY_TYPE_EPSILON));
  assert_true(is_equal_ndarray(var_a5->grad, a5_grad, NDARRAY_TYPE_EPSILON));
  assert_true(is_equal_ndarray(var_b0->grad, b0_grad, NDARRAY_TYPE_EPSILON));
  assert_true(is_equal_ndarray(var_b1->grad, b1_grad, NDARRAY_TYPE_EPSILON));
  assert_true(is_equal_ndarray(var_b2->grad, b2_grad, NDARRAY_TYPE_EPSILON));
  assert_true(is_equal_ndarray(var_b3->grad, b3_grad, NDARRAY_TYPE_EPSILON));
  assert_true(is_equal_ndarray(var_b4->grad, b4_grad, NDARRAY_TYPE_EPSILON));
  assert_true(is_equal_ndarray(var_b5->grad, b5_grad, NDARRAY_TYPE_EPSILON));

  free_ndarray(&a0);
  free_ndarray(&a1);
  free_ndarray(&a2);
  free_ndarray(&a3);
  free_ndarray(&a4);
  free_ndarray(&a5);
  free_ndarray(&b0);
  free_ndarray(&b1);
  free_ndarray(&b2);
  free_ndarray(&b3);
  free_ndarray(&b4);
  free_ndarray(&b5);
  free_ndarray(&c0);
  free_ndarray(&c1);
  free_ndarray(&c2);
  free_ndarray(&c3);
  free_ndarray(&c4);
  free_ndarray(&c5);
  free_ndarray(&a0_grad);
  free_ndarray(&a1_grad);
  free_ndarray(&a2_grad);
  free_ndarray(&a3_grad);
  free_ndarray(&a4_grad);
  free_ndarray(&a5_grad);
  free_ndarray(&b0_grad);
  free_ndarray(&b1_grad);
  free_ndarray(&b2_grad);
  free_ndarray(&b3_grad);
  free_ndarray(&b4_grad);
  free_ndarray(&b5_grad);

  free_graph_variable(&var_c0_hat);
  free_graph_variable(&var_c1_hat);
  free_graph_variable(&var_c2_hat);
  free_graph_variable(&var_c3_hat);
  free_graph_variable(&var_c4_hat);
  free_graph_variable(&var_c5_hat);
}

static void test_subtract_variable(void **state) {
  (void)state;
  ndarray *a0 =
      read_ndarray("./test_data/test_variable/test_subtract_variable/a0.txt");
  ndarray *a1 =
      read_ndarray("./test_data/test_variable/test_subtract_variable/a1.txt");
  ndarray *a2 =
      read_ndarray("./test_data/test_variable/test_subtract_variable/a2.txt");
  ndarray *a3 =
      read_ndarray("./test_data/test_variable/test_subtract_variable/a3.txt");
  ndarray *a4 =
      read_ndarray("./test_data/test_variable/test_subtract_variable/a4.txt");
  ndarray *a5 =
      read_ndarray("./test_data/test_variable/test_subtract_variable/a5.txt");
  ndarray *b0 =
      read_ndarray("./test_data/test_variable/test_subtract_variable/b0.txt");
  ndarray *b1 =
      read_ndarray("./test_data/test_variable/test_subtract_variable/b1.txt");
  ndarray *b2 =
      read_ndarray("./test_data/test_variable/test_subtract_variable/b2.txt");
  ndarray *b3 =
      read_ndarray("./test_data/test_variable/test_subtract_variable/b3.txt");
  ndarray *b4 =
      read_ndarray("./test_data/test_variable/test_subtract_variable/b4.txt");
  ndarray *b5 =
      read_ndarray("./test_data/test_variable/test_subtract_variable/b5.txt");
  ndarray *c0 =
      read_ndarray("./test_data/test_variable/test_subtract_variable/c0.txt");
  ndarray *c1 =
      read_ndarray("./test_data/test_variable/test_subtract_variable/c1.txt");
  ndarray *c2 =
      read_ndarray("./test_data/test_variable/test_subtract_variable/c2.txt");
  ndarray *c3 =
      read_ndarray("./test_data/test_variable/test_subtract_variable/c3.txt");
  ndarray *c4 =
      read_ndarray("./test_data/test_variable/test_subtract_variable/c4.txt");
  ndarray *c5 =
      read_ndarray("./test_data/test_variable/test_subtract_variable/c5.txt");
  ndarray *d0 =
      read_ndarray("./test_data/test_variable/test_subtract_variable/d0.txt");
  ndarray *d1 =
      read_ndarray("./test_data/test_variable/test_subtract_variable/d1.txt");
  ndarray *d2 =
      read_ndarray("./test_data/test_variable/test_subtract_variable/d2.txt");
  ndarray *d3 =
      read_ndarray("./test_data/test_variable/test_subtract_variable/d3.txt");
  ndarray *d4 =
      read_ndarray("./test_data/test_variable/test_subtract_variable/d4.txt");
  ndarray *d5 =
      read_ndarray("./test_data/test_variable/test_subtract_variable/d5.txt");
  ndarray *a0_grad = read_ndarray(
      "./test_data/test_variable/test_subtract_variable/a0_grad.txt");
  ndarray *a1_grad = read_ndarray(
      "./test_data/test_variable/test_subtract_variable/a1_grad.txt");
  ndarray *a2_grad = read_ndarray(
      "./test_data/test_variable/test_subtract_variable/a2_grad.txt");
  ndarray *a3_grad = read_ndarray(
      "./test_data/test_variable/test_subtract_variable/a3_grad.txt");
  ndarray *a4_grad = read_ndarray(
      "./test_data/test_variable/test_subtract_variable/a4_grad.txt");
  ndarray *a5_grad = read_ndarray(
      "./test_data/test_variable/test_subtract_variable/a5_grad.txt");
  ndarray *b0_grad = read_ndarray(
      "./test_data/test_variable/test_subtract_variable/b0_grad.txt");
  ndarray *b1_grad = read_ndarray(
      "./test_data/test_variable/test_subtract_variable/b1_grad.txt");
  ndarray *b2_grad = read_ndarray(
      "./test_data/test_variable/test_subtract_variable/b2_grad.txt");
  ndarray *b3_grad = read_ndarray(
      "./test_data/test_variable/test_subtract_variable/b3_grad.txt");
  ndarray *b4_grad = read_ndarray(
      "./test_data/test_variable/test_subtract_variable/b4_grad.txt");
  ndarray *b5_grad = read_ndarray(
      "./test_data/test_variable/test_subtract_variable/b5_grad.txt");

  variable *var_a0 = new_variable(a0);
  variable *var_a1 = new_variable(a1);
  variable *var_a2 = new_variable(a2);
  variable *var_a3 = new_variable(a3);
  variable *var_a4 = new_variable(a4);
  variable *var_a5 = new_variable(a5);
  variable *var_b0 = new_variable(b0);
  variable *var_b1 = new_variable(b1);
  variable *var_b2 = new_variable(b2);
  variable *var_b3 = new_variable(b3);
  variable *var_b4 = new_variable(b4);
  variable *var_b5 = new_variable(b5);
  variable *var_c0_hat = subtract_variable(var_a0, var_b0);
  variable *var_c1_hat = subtract_variable(var_a1, var_b1);
  variable *var_c2_hat = subtract_variable(var_a2, var_b2);
  variable *var_c3_hat = subtract_variable(var_a3, var_b3);
  variable *var_c4_hat = subtract_variable(var_a4, var_b4);
  variable *var_c5_hat = subtract_variable(var_a5, var_b5);
  free_ndarray(&(var_c0_hat->grad));
  free_ndarray(&(var_c1_hat->grad));
  free_ndarray(&(var_c2_hat->grad));
  free_ndarray(&(var_c3_hat->grad));
  free_ndarray(&(var_c4_hat->grad));
  free_ndarray(&(var_c5_hat->grad));
  var_c0_hat->grad = d0;
  var_c1_hat->grad = d1;
  var_c2_hat->grad = d2;
  var_c3_hat->grad = d3;
  var_c4_hat->grad = d4;
  var_c5_hat->grad = d5;
  backward_variable(var_c0_hat);
  backward_variable(var_c1_hat);
  backward_variable(var_c2_hat);
  backward_variable(var_c3_hat);
  backward_variable(var_c4_hat);
  backward_variable(var_c5_hat);

  assert_true(is_equal_ndarray(var_c0_hat->val, c0, NDARRAY_TYPE_EPSILON));
  assert_true(is_equal_ndarray(var_c1_hat->val, c1, NDARRAY_TYPE_EPSILON));
  assert_true(is_equal_ndarray(var_c2_hat->val, c2, NDARRAY_TYPE_EPSILON));
  assert_true(is_equal_ndarray(var_c3_hat->val, c3, NDARRAY_TYPE_EPSILON));
  assert_true(is_equal_ndarray(var_c4_hat->val, c4, NDARRAY_TYPE_EPSILON));
  assert_true(is_equal_ndarray(var_c5_hat->val, c5, NDARRAY_TYPE_EPSILON));
  assert_true(is_equal_ndarray(var_a0->grad, a0_grad, NDARRAY_TYPE_EPSILON));
  assert_true(is_equal_ndarray(var_a1->grad, a1_grad, NDARRAY_TYPE_EPSILON));
  assert_true(is_equal_ndarray(var_a2->grad, a2_grad, NDARRAY_TYPE_EPSILON));
  assert_true(is_equal_ndarray(var_a3->grad, a3_grad, NDARRAY_TYPE_EPSILON));
  assert_true(is_equal_ndarray(var_a4->grad, a4_grad, NDARRAY_TYPE_EPSILON));
  assert_true(is_equal_ndarray(var_a5->grad, a5_grad, NDARRAY_TYPE_EPSILON));
  assert_true(is_equal_ndarray(var_b0->grad, b0_grad, NDARRAY_TYPE_EPSILON));
  assert_true(is_equal_ndarray(var_b1->grad, b1_grad, NDARRAY_TYPE_EPSILON));
  assert_true(is_equal_ndarray(var_b2->grad, b2_grad, NDARRAY_TYPE_EPSILON));
  assert_true(is_equal_ndarray(var_b3->grad, b3_grad, NDARRAY_TYPE_EPSILON));
  assert_true(is_equal_ndarray(var_b4->grad, b4_grad, NDARRAY_TYPE_EPSILON));
  assert_true(is_equal_ndarray(var_b5->grad, b5_grad, NDARRAY_TYPE_EPSILON));

  free_ndarray(&a0);
  free_ndarray(&a1);
  free_ndarray(&a2);
  free_ndarray(&a3);
  free_ndarray(&a4);
  free_ndarray(&a5);
  free_ndarray(&b0);
  free_ndarray(&b1);
  free_ndarray(&b2);
  free_ndarray(&b3);
  free_ndarray(&b4);
  free_ndarray(&b5);
  free_ndarray(&c0);
  free_ndarray(&c1);
  free_ndarray(&c2);
  free_ndarray(&c3);
  free_ndarray(&c4);
  free_ndarray(&c5);
  free_ndarray(&a0_grad);
  free_ndarray(&a1_grad);
  free_ndarray(&a2_grad);
  free_ndarray(&a3_grad);
  free_ndarray(&a4_grad);
  free_ndarray(&a5_grad);
  free_ndarray(&b0_grad);
  free_ndarray(&b1_grad);
  free_ndarray(&b2_grad);
  free_ndarray(&b3_grad);
  free_ndarray(&b4_grad);
  free_ndarray(&b5_grad);

  free_graph_variable(&var_c0_hat);
  free_graph_variable(&var_c1_hat);
  free_graph_variable(&var_c2_hat);
  free_graph_variable(&var_c3_hat);
  free_graph_variable(&var_c4_hat);
  free_graph_variable(&var_c5_hat);
}

static void test_multiply_variable(void **state) {
  (void)state;
  ndarray *a0 =
      read_ndarray("./test_data/test_variable/test_multiply_variable/a0.txt");
  ndarray *a1 =
      read_ndarray("./test_data/test_variable/test_multiply_variable/a1.txt");
  ndarray *a2 =
      read_ndarray("./test_data/test_variable/test_multiply_variable/a2.txt");
  ndarray *a3 =
      read_ndarray("./test_data/test_variable/test_multiply_variable/a3.txt");
  ndarray *a4 =
      read_ndarray("./test_data/test_variable/test_multiply_variable/a4.txt");
  ndarray *a5 =
      read_ndarray("./test_data/test_variable/test_multiply_variable/a5.txt");
  ndarray *b0 =
      read_ndarray("./test_data/test_variable/test_multiply_variable/b0.txt");
  ndarray *b1 =
      read_ndarray("./test_data/test_variable/test_multiply_variable/b1.txt");
  ndarray *b2 =
      read_ndarray("./test_data/test_variable/test_multiply_variable/b2.txt");
  ndarray *b3 =
      read_ndarray("./test_data/test_variable/test_multiply_variable/b3.txt");
  ndarray *b4 =
      read_ndarray("./test_data/test_variable/test_multiply_variable/b4.txt");
  ndarray *b5 =
      read_ndarray("./test_data/test_variable/test_multiply_variable/b5.txt");
  ndarray *c0 =
      read_ndarray("./test_data/test_variable/test_multiply_variable/c0.txt");
  ndarray *c1 =
      read_ndarray("./test_data/test_variable/test_multiply_variable/c1.txt");
  ndarray *c2 =
      read_ndarray("./test_data/test_variable/test_multiply_variable/c2.txt");
  ndarray *c3 =
      read_ndarray("./test_data/test_variable/test_multiply_variable/c3.txt");
  ndarray *c4 =
      read_ndarray("./test_data/test_variable/test_multiply_variable/c4.txt");
  ndarray *c5 =
      read_ndarray("./test_data/test_variable/test_multiply_variable/c5.txt");
  ndarray *d0 =
      read_ndarray("./test_data/test_variable/test_multiply_variable/d0.txt");
  ndarray *d1 =
      read_ndarray("./test_data/test_variable/test_multiply_variable/d1.txt");
  ndarray *d2 =
      read_ndarray("./test_data/test_variable/test_multiply_variable/d2.txt");
  ndarray *d3 =
      read_ndarray("./test_data/test_variable/test_multiply_variable/d3.txt");
  ndarray *d4 =
      read_ndarray("./test_data/test_variable/test_multiply_variable/d4.txt");
  ndarray *d5 =
      read_ndarray("./test_data/test_variable/test_multiply_variable/d5.txt");
  ndarray *a0_grad = read_ndarray(
      "./test_data/test_variable/test_multiply_variable/a0_grad.txt");
  ndarray *a1_grad = read_ndarray(
      "./test_data/test_variable/test_multiply_variable/a1_grad.txt");
  ndarray *a2_grad = read_ndarray(
      "./test_data/test_variable/test_multiply_variable/a2_grad.txt");
  ndarray *a3_grad = read_ndarray(
      "./test_data/test_variable/test_multiply_variable/a3_grad.txt");
  ndarray *a4_grad = read_ndarray(
      "./test_data/test_variable/test_multiply_variable/a4_grad.txt");
  ndarray *a5_grad = read_ndarray(
      "./test_data/test_variable/test_multiply_variable/a5_grad.txt");
  ndarray *b0_grad = read_ndarray(
      "./test_data/test_variable/test_multiply_variable/b0_grad.txt");
  ndarray *b1_grad = read_ndarray(
      "./test_data/test_variable/test_multiply_variable/b1_grad.txt");
  ndarray *b2_grad = read_ndarray(
      "./test_data/test_variable/test_multiply_variable/b2_grad.txt");
  ndarray *b3_grad = read_ndarray(
      "./test_data/test_variable/test_multiply_variable/b3_grad.txt");
  ndarray *b4_grad = read_ndarray(
      "./test_data/test_variable/test_multiply_variable/b4_grad.txt");
  ndarray *b5_grad = read_ndarray(
      "./test_data/test_variable/test_multiply_variable/b5_grad.txt");

  variable *var_a0 = new_variable(a0);
  variable *var_a1 = new_variable(a1);
  variable *var_a2 = new_variable(a2);
  variable *var_a3 = new_variable(a3);
  variable *var_a4 = new_variable(a4);
  variable *var_a5 = new_variable(a5);
  variable *var_b0 = new_variable(b0);
  variable *var_b1 = new_variable(b1);
  variable *var_b2 = new_variable(b2);
  variable *var_b3 = new_variable(b3);
  variable *var_b4 = new_variable(b4);
  variable *var_b5 = new_variable(b5);
  variable *var_c0_hat = multiply_variable(var_a0, var_b0);
  variable *var_c1_hat = multiply_variable(var_a1, var_b1);
  variable *var_c2_hat = multiply_variable(var_a2, var_b2);
  variable *var_c3_hat = multiply_variable(var_a3, var_b3);
  variable *var_c4_hat = multiply_variable(var_a4, var_b4);
  variable *var_c5_hat = multiply_variable(var_a5, var_b5);
  free_ndarray(&(var_c0_hat->grad));
  free_ndarray(&(var_c1_hat->grad));
  free_ndarray(&(var_c2_hat->grad));
  free_ndarray(&(var_c3_hat->grad));
  free_ndarray(&(var_c4_hat->grad));
  free_ndarray(&(var_c5_hat->grad));
  var_c0_hat->grad = d0;
  var_c1_hat->grad = d1;
  var_c2_hat->grad = d2;
  var_c3_hat->grad = d3;
  var_c4_hat->grad = d4;
  var_c5_hat->grad = d5;
  backward_variable(var_c0_hat);
  backward_variable(var_c1_hat);
  backward_variable(var_c2_hat);
  backward_variable(var_c3_hat);
  backward_variable(var_c4_hat);
  backward_variable(var_c5_hat);

  assert_true(is_equal_ndarray(var_c0_hat->val, c0, NDARRAY_TYPE_EPSILON));
  assert_true(is_equal_ndarray(var_c1_hat->val, c1, NDARRAY_TYPE_EPSILON));
  assert_true(is_equal_ndarray(var_c2_hat->val, c2, NDARRAY_TYPE_EPSILON));
  assert_true(is_equal_ndarray(var_c3_hat->val, c3, NDARRAY_TYPE_EPSILON));
  assert_true(is_equal_ndarray(var_c4_hat->val, c4, NDARRAY_TYPE_EPSILON));
  assert_true(is_equal_ndarray(var_c5_hat->val, c5, NDARRAY_TYPE_EPSILON));
  assert_true(is_equal_ndarray(var_a0->grad, a0_grad, NDARRAY_TYPE_EPSILON));
  assert_true(is_equal_ndarray(var_a1->grad, a1_grad, NDARRAY_TYPE_EPSILON));
  assert_true(is_equal_ndarray(var_a2->grad, a2_grad, NDARRAY_TYPE_EPSILON));
  assert_true(is_equal_ndarray(var_a3->grad, a3_grad, NDARRAY_TYPE_EPSILON));
  assert_true(is_equal_ndarray(var_a4->grad, a4_grad, NDARRAY_TYPE_EPSILON));
  assert_true(is_equal_ndarray(var_a5->grad, a5_grad, NDARRAY_TYPE_EPSILON));
  assert_true(is_equal_ndarray(var_b0->grad, b0_grad, NDARRAY_TYPE_EPSILON));
  assert_true(is_equal_ndarray(var_b1->grad, b1_grad, NDARRAY_TYPE_EPSILON));
  assert_true(is_equal_ndarray(var_b2->grad, b2_grad, NDARRAY_TYPE_EPSILON));
  assert_true(is_equal_ndarray(var_b3->grad, b3_grad, NDARRAY_TYPE_EPSILON));
  assert_true(is_equal_ndarray(var_b4->grad, b4_grad, NDARRAY_TYPE_EPSILON));
  assert_true(is_equal_ndarray(var_b5->grad, b5_grad, NDARRAY_TYPE_EPSILON));

  free_ndarray(&a0);
  free_ndarray(&a1);
  free_ndarray(&a2);
  free_ndarray(&a3);
  free_ndarray(&a4);
  free_ndarray(&a5);
  free_ndarray(&b0);
  free_ndarray(&b1);
  free_ndarray(&b2);
  free_ndarray(&b3);
  free_ndarray(&b4);
  free_ndarray(&b5);
  free_ndarray(&c0);
  free_ndarray(&c1);
  free_ndarray(&c2);
  free_ndarray(&c3);
  free_ndarray(&c4);
  free_ndarray(&c5);
  free_ndarray(&a0_grad);
  free_ndarray(&a1_grad);
  free_ndarray(&a2_grad);
  free_ndarray(&a3_grad);
  free_ndarray(&a4_grad);
  free_ndarray(&a5_grad);
  free_ndarray(&b0_grad);
  free_ndarray(&b1_grad);
  free_ndarray(&b2_grad);
  free_ndarray(&b3_grad);
  free_ndarray(&b4_grad);
  free_ndarray(&b5_grad);

  free_graph_variable(&var_c0_hat);
  free_graph_variable(&var_c1_hat);
  free_graph_variable(&var_c2_hat);
  free_graph_variable(&var_c3_hat);
  free_graph_variable(&var_c4_hat);
  free_graph_variable(&var_c5_hat);
}

static void test_divide_variable(void **state) {
  (void)state;
  ndarray *a0 =
      read_ndarray("./test_data/test_variable/test_divide_variable/a0.txt");
  ndarray *a1 =
      read_ndarray("./test_data/test_variable/test_divide_variable/a1.txt");
  ndarray *a2 =
      read_ndarray("./test_data/test_variable/test_divide_variable/a2.txt");
  ndarray *a3 =
      read_ndarray("./test_data/test_variable/test_divide_variable/a3.txt");
  ndarray *a4 =
      read_ndarray("./test_data/test_variable/test_divide_variable/a4.txt");
  ndarray *a5 =
      read_ndarray("./test_data/test_variable/test_divide_variable/a5.txt");
  ndarray *b0 =
      read_ndarray("./test_data/test_variable/test_divide_variable/b0.txt");
  ndarray *b1 =
      read_ndarray("./test_data/test_variable/test_divide_variable/b1.txt");
  ndarray *b2 =
      read_ndarray("./test_data/test_variable/test_divide_variable/b2.txt");
  ndarray *b3 =
      read_ndarray("./test_data/test_variable/test_divide_variable/b3.txt");
  ndarray *b4 =
      read_ndarray("./test_data/test_variable/test_divide_variable/b4.txt");
  ndarray *b5 =
      read_ndarray("./test_data/test_variable/test_divide_variable/b5.txt");
  ndarray *c0 =
      read_ndarray("./test_data/test_variable/test_divide_variable/c0.txt");
  ndarray *c1 =
      read_ndarray("./test_data/test_variable/test_divide_variable/c1.txt");
  ndarray *c2 =
      read_ndarray("./test_data/test_variable/test_divide_variable/c2.txt");
  ndarray *c3 =
      read_ndarray("./test_data/test_variable/test_divide_variable/c3.txt");
  ndarray *c4 =
      read_ndarray("./test_data/test_variable/test_divide_variable/c4.txt");
  ndarray *c5 =
      read_ndarray("./test_data/test_variable/test_divide_variable/c5.txt");
  ndarray *d0 =
      read_ndarray("./test_data/test_variable/test_divide_variable/d0.txt");
  ndarray *d1 =
      read_ndarray("./test_data/test_variable/test_divide_variable/d1.txt");
  ndarray *d2 =
      read_ndarray("./test_data/test_variable/test_divide_variable/d2.txt");
  ndarray *d3 =
      read_ndarray("./test_data/test_variable/test_divide_variable/d3.txt");
  ndarray *d4 =
      read_ndarray("./test_data/test_variable/test_divide_variable/d4.txt");
  ndarray *d5 =
      read_ndarray("./test_data/test_variable/test_divide_variable/d5.txt");
  ndarray *a0_grad = read_ndarray(
      "./test_data/test_variable/test_divide_variable/a0_grad.txt");
  ndarray *a1_grad = read_ndarray(
      "./test_data/test_variable/test_divide_variable/a1_grad.txt");
  ndarray *a2_grad = read_ndarray(
      "./test_data/test_variable/test_divide_variable/a2_grad.txt");
  ndarray *a3_grad = read_ndarray(
      "./test_data/test_variable/test_divide_variable/a3_grad.txt");
  ndarray *a4_grad = read_ndarray(
      "./test_data/test_variable/test_divide_variable/a4_grad.txt");
  ndarray *a5_grad = read_ndarray(
      "./test_data/test_variable/test_divide_variable/a5_grad.txt");
  ndarray *b0_grad = read_ndarray(
      "./test_data/test_variable/test_divide_variable/b0_grad.txt");
  ndarray *b1_grad = read_ndarray(
      "./test_data/test_variable/test_divide_variable/b1_grad.txt");
  ndarray *b2_grad = read_ndarray(
      "./test_data/test_variable/test_divide_variable/b2_grad.txt");
  ndarray *b3_grad = read_ndarray(
      "./test_data/test_variable/test_divide_variable/b3_grad.txt");
  ndarray *b4_grad = read_ndarray(
      "./test_data/test_variable/test_divide_variable/b4_grad.txt");
  ndarray *b5_grad = read_ndarray(
      "./test_data/test_variable/test_divide_variable/b5_grad.txt");

  variable *var_a0 = new_variable(a0);
  variable *var_a1 = new_variable(a1);
  variable *var_a2 = new_variable(a2);
  variable *var_a3 = new_variable(a3);
  variable *var_a4 = new_variable(a4);
  variable *var_a5 = new_variable(a5);
  variable *var_b0 = new_variable(b0);
  variable *var_b1 = new_variable(b1);
  variable *var_b2 = new_variable(b2);
  variable *var_b3 = new_variable(b3);
  variable *var_b4 = new_variable(b4);
  variable *var_b5 = new_variable(b5);
  variable *var_c0_hat = divide_variable(var_a0, var_b0);
  variable *var_c1_hat = divide_variable(var_a1, var_b1);
  variable *var_c2_hat = divide_variable(var_a2, var_b2);
  variable *var_c3_hat = divide_variable(var_a3, var_b3);
  variable *var_c4_hat = divide_variable(var_a4, var_b4);
  variable *var_c5_hat = divide_variable(var_a5, var_b5);
  free_ndarray(&(var_c0_hat->grad));
  free_ndarray(&(var_c1_hat->grad));
  free_ndarray(&(var_c2_hat->grad));
  free_ndarray(&(var_c3_hat->grad));
  free_ndarray(&(var_c4_hat->grad));
  free_ndarray(&(var_c5_hat->grad));
  var_c0_hat->grad = d0;
  var_c1_hat->grad = d1;
  var_c2_hat->grad = d2;
  var_c3_hat->grad = d3;
  var_c4_hat->grad = d4;
  var_c5_hat->grad = d5;
  backward_variable(var_c0_hat);
  backward_variable(var_c1_hat);
  backward_variable(var_c2_hat);
  backward_variable(var_c3_hat);
  backward_variable(var_c4_hat);
  backward_variable(var_c5_hat);

  assert_true(is_equal_ndarray(var_c0_hat->val, c0, NDARRAY_TYPE_EPSILON));
  assert_true(is_equal_ndarray(var_c1_hat->val, c1, NDARRAY_TYPE_EPSILON));
  assert_true(is_equal_ndarray(var_c2_hat->val, c2, NDARRAY_TYPE_EPSILON));
  assert_true(is_equal_ndarray(var_c3_hat->val, c3, NDARRAY_TYPE_EPSILON));
  assert_true(is_equal_ndarray(var_c4_hat->val, c4, NDARRAY_TYPE_EPSILON));
  assert_true(is_equal_ndarray(var_c5_hat->val, c5, NDARRAY_TYPE_EPSILON));
  assert_true(is_equal_ndarray(var_a0->grad, a0_grad, NDARRAY_TYPE_EPSILON));
  assert_true(is_equal_ndarray(var_a1->grad, a1_grad, NDARRAY_TYPE_EPSILON));
  assert_true(is_equal_ndarray(var_a2->grad, a2_grad, NDARRAY_TYPE_EPSILON));
  assert_true(is_equal_ndarray(var_a3->grad, a3_grad, NDARRAY_TYPE_EPSILON));
  assert_true(is_equal_ndarray(var_a4->grad, a4_grad, NDARRAY_TYPE_EPSILON));
  assert_true(is_equal_ndarray(var_a5->grad, a5_grad, NDARRAY_TYPE_EPSILON));
  assert_true(is_equal_ndarray(var_b0->grad, b0_grad, NDARRAY_TYPE_EPSILON));
  assert_true(is_equal_ndarray(var_b1->grad, b1_grad, NDARRAY_TYPE_EPSILON));
  assert_true(is_equal_ndarray(var_b2->grad, b2_grad, NDARRAY_TYPE_EPSILON));
  assert_true(is_equal_ndarray(var_b3->grad, b3_grad, NDARRAY_TYPE_EPSILON));
  assert_true(is_equal_ndarray(var_b4->grad, b4_grad, NDARRAY_TYPE_EPSILON));
  assert_true(is_equal_ndarray(var_b5->grad, b5_grad, NDARRAY_TYPE_EPSILON));

  free_ndarray(&a0);
  free_ndarray(&a1);
  free_ndarray(&a2);
  free_ndarray(&a3);
  free_ndarray(&a4);
  free_ndarray(&a5);
  free_ndarray(&b0);
  free_ndarray(&b1);
  free_ndarray(&b2);
  free_ndarray(&b3);
  free_ndarray(&b4);
  free_ndarray(&b5);
  free_ndarray(&c0);
  free_ndarray(&c1);
  free_ndarray(&c2);
  free_ndarray(&c3);
  free_ndarray(&c4);
  free_ndarray(&c5);
  free_ndarray(&a0_grad);
  free_ndarray(&a1_grad);
  free_ndarray(&a2_grad);
  free_ndarray(&a3_grad);
  free_ndarray(&a4_grad);
  free_ndarray(&a5_grad);
  free_ndarray(&b0_grad);
  free_ndarray(&b1_grad);
  free_ndarray(&b2_grad);
  free_ndarray(&b3_grad);
  free_ndarray(&b4_grad);
  free_ndarray(&b5_grad);

  free_graph_variable(&var_c0_hat);
  free_graph_variable(&var_c1_hat);
  free_graph_variable(&var_c2_hat);
  free_graph_variable(&var_c3_hat);
  free_graph_variable(&var_c4_hat);
  free_graph_variable(&var_c5_hat);
}

static void test_power_variable(void **state) {
  (void)state;
  ndarray *a0 =
      read_ndarray("./test_data/test_variable/test_power_variable/a0.txt");
  ndarray *a1 =
      read_ndarray("./test_data/test_variable/test_power_variable/a1.txt");
  ndarray *a2 =
      read_ndarray("./test_data/test_variable/test_power_variable/a2.txt");
  ndarray *a3 =
      read_ndarray("./test_data/test_variable/test_power_variable/a3.txt");
  ndarray *a4 =
      read_ndarray("./test_data/test_variable/test_power_variable/a4.txt");
  ndarray *a5 =
      read_ndarray("./test_data/test_variable/test_power_variable/a5.txt");
  ndarray *b0 =
      read_ndarray("./test_data/test_variable/test_power_variable/b0.txt");
  ndarray *b1 =
      read_ndarray("./test_data/test_variable/test_power_variable/b1.txt");
  ndarray *b2 =
      read_ndarray("./test_data/test_variable/test_power_variable/b2.txt");
  ndarray *b3 =
      read_ndarray("./test_data/test_variable/test_power_variable/b3.txt");
  ndarray *b4 =
      read_ndarray("./test_data/test_variable/test_power_variable/b4.txt");
  ndarray *b5 =
      read_ndarray("./test_data/test_variable/test_power_variable/b5.txt");
  ndarray *c0 =
      read_ndarray("./test_data/test_variable/test_power_variable/c0.txt");
  ndarray *c1 =
      read_ndarray("./test_data/test_variable/test_power_variable/c1.txt");
  ndarray *c2 =
      read_ndarray("./test_data/test_variable/test_power_variable/c2.txt");
  ndarray *c3 =
      read_ndarray("./test_data/test_variable/test_power_variable/c3.txt");
  ndarray *c4 =
      read_ndarray("./test_data/test_variable/test_power_variable/c4.txt");
  ndarray *c5 =
      read_ndarray("./test_data/test_variable/test_power_variable/c5.txt");
  ndarray *d0 =
      read_ndarray("./test_data/test_variable/test_power_variable/d0.txt");
  ndarray *d1 =
      read_ndarray("./test_data/test_variable/test_power_variable/d1.txt");
  ndarray *d2 =
      read_ndarray("./test_data/test_variable/test_power_variable/d2.txt");
  ndarray *d3 =
      read_ndarray("./test_data/test_variable/test_power_variable/d3.txt");
  ndarray *d4 =
      read_ndarray("./test_data/test_variable/test_power_variable/d4.txt");
  ndarray *d5 =
      read_ndarray("./test_data/test_variable/test_power_variable/d5.txt");
  ndarray *a0_grad =
      read_ndarray("./test_data/test_variable/test_power_variable/a0_grad.txt");
  ndarray *a1_grad =
      read_ndarray("./test_data/test_variable/test_power_variable/a1_grad.txt");
  ndarray *a2_grad =
      read_ndarray("./test_data/test_variable/test_power_variable/a2_grad.txt");
  ndarray *a3_grad =
      read_ndarray("./test_data/test_variable/test_power_variable/a3_grad.txt");
  ndarray *a4_grad =
      read_ndarray("./test_data/test_variable/test_power_variable/a4_grad.txt");
  ndarray *a5_grad =
      read_ndarray("./test_data/test_variable/test_power_variable/a5_grad.txt");
  ndarray *b0_grad =
      read_ndarray("./test_data/test_variable/test_power_variable/b0_grad.txt");
  ndarray *b1_grad =
      read_ndarray("./test_data/test_variable/test_power_variable/b1_grad.txt");
  ndarray *b2_grad =
      read_ndarray("./test_data/test_variable/test_power_variable/b2_grad.txt");
  ndarray *b3_grad =
      read_ndarray("./test_data/test_variable/test_power_variable/b3_grad.txt");
  ndarray *b4_grad =
      read_ndarray("./test_data/test_variable/test_power_variable/b4_grad.txt");
  ndarray *b5_grad =
      read_ndarray("./test_data/test_variable/test_power_variable/b5_grad.txt");

  variable *var_a0 = new_variable(a0);
  variable *var_a1 = new_variable(a1);
  variable *var_a2 = new_variable(a2);
  variable *var_a3 = new_variable(a3);
  variable *var_a4 = new_variable(a4);
  variable *var_a5 = new_variable(a5);
  variable *var_b0 = new_variable(b0);
  variable *var_b1 = new_variable(b1);
  variable *var_b2 = new_variable(b2);
  variable *var_b3 = new_variable(b3);
  variable *var_b4 = new_variable(b4);
  variable *var_b5 = new_variable(b5);
  variable *var_c0_hat = power_variable(var_a0, var_b0);
  variable *var_c1_hat = power_variable(var_a1, var_b1);
  variable *var_c2_hat = power_variable(var_a2, var_b2);
  variable *var_c3_hat = power_variable(var_a3, var_b3);
  variable *var_c4_hat = power_variable(var_a4, var_b4);
  variable *var_c5_hat = power_variable(var_a5, var_b5);
  free_ndarray(&(var_c0_hat->grad));
  free_ndarray(&(var_c1_hat->grad));
  free_ndarray(&(var_c2_hat->grad));
  free_ndarray(&(var_c3_hat->grad));
  free_ndarray(&(var_c4_hat->grad));
  free_ndarray(&(var_c5_hat->grad));
  var_c0_hat->grad = d0;
  var_c1_hat->grad = d1;
  var_c2_hat->grad = d2;
  var_c3_hat->grad = d3;
  var_c4_hat->grad = d4;
  var_c5_hat->grad = d5;
  backward_variable(var_c0_hat);
  backward_variable(var_c1_hat);
  backward_variable(var_c2_hat);
  backward_variable(var_c3_hat);
  backward_variable(var_c4_hat);
  backward_variable(var_c5_hat);

  assert_true(is_equal_ndarray(var_c0_hat->val, c0, NDARRAY_TYPE_EPSILON));
  assert_true(is_equal_ndarray(var_c1_hat->val, c1, NDARRAY_TYPE_EPSILON));
  assert_true(is_equal_ndarray(var_c2_hat->val, c2, NDARRAY_TYPE_EPSILON));
  assert_true(is_equal_ndarray(var_c3_hat->val, c3, NDARRAY_TYPE_EPSILON));
  assert_true(is_equal_ndarray(var_c4_hat->val, c4, NDARRAY_TYPE_EPSILON));
  assert_true(is_equal_ndarray(var_c5_hat->val, c5, NDARRAY_TYPE_EPSILON));
  assert_true(is_equal_ndarray(var_a0->grad, a0_grad, NDARRAY_TYPE_EPSILON));
  assert_true(is_equal_ndarray(var_a1->grad, a1_grad, NDARRAY_TYPE_EPSILON));
  assert_true(is_equal_ndarray(var_a2->grad, a2_grad, NDARRAY_TYPE_EPSILON));
  assert_true(is_equal_ndarray(var_a3->grad, a3_grad, NDARRAY_TYPE_EPSILON));
  assert_true(is_equal_ndarray(var_a4->grad, a4_grad, NDARRAY_TYPE_EPSILON));
  assert_true(is_equal_ndarray(var_a5->grad, a5_grad, NDARRAY_TYPE_EPSILON));
  assert_true(is_equal_ndarray(var_b0->grad, b0_grad, NDARRAY_TYPE_EPSILON));
  assert_true(is_equal_ndarray(var_b1->grad, b1_grad, NDARRAY_TYPE_EPSILON));
  assert_true(is_equal_ndarray(var_b2->grad, b2_grad, NDARRAY_TYPE_EPSILON));
  assert_true(is_equal_ndarray(var_b3->grad, b3_grad, NDARRAY_TYPE_EPSILON));
  assert_true(is_equal_ndarray(var_b4->grad, b4_grad, NDARRAY_TYPE_EPSILON));
  assert_true(is_equal_ndarray(var_b5->grad, b5_grad, NDARRAY_TYPE_EPSILON));

  free_ndarray(&a0);
  free_ndarray(&a1);
  free_ndarray(&a2);
  free_ndarray(&a3);
  free_ndarray(&a4);
  free_ndarray(&a5);
  free_ndarray(&b0);
  free_ndarray(&b1);
  free_ndarray(&b2);
  free_ndarray(&b3);
  free_ndarray(&b4);
  free_ndarray(&b5);
  free_ndarray(&c0);
  free_ndarray(&c1);
  free_ndarray(&c2);
  free_ndarray(&c3);
  free_ndarray(&c4);
  free_ndarray(&c5);
  free_ndarray(&a0_grad);
  free_ndarray(&a1_grad);
  free_ndarray(&a2_grad);
  free_ndarray(&a3_grad);
  free_ndarray(&a4_grad);
  free_ndarray(&a5_grad);
  free_ndarray(&b0_grad);
  free_ndarray(&b1_grad);
  free_ndarray(&b2_grad);
  free_ndarray(&b3_grad);
  free_ndarray(&b4_grad);
  free_ndarray(&b5_grad);

  free_graph_variable(&var_c0_hat);
  free_graph_variable(&var_c1_hat);
  free_graph_variable(&var_c2_hat);
  free_graph_variable(&var_c3_hat);
  free_graph_variable(&var_c4_hat);
  free_graph_variable(&var_c5_hat);
}

static void test_exp_variable(void **state) {
  (void)state;
  ndarray *x0 =
      read_ndarray("./test_data/test_variable/test_exp_variable/x0.txt");
  ndarray *x1 =
      read_ndarray("./test_data/test_variable/test_exp_variable/x1.txt");
  ndarray *x2 =
      read_ndarray("./test_data/test_variable/test_exp_variable/x2.txt");
  ndarray *y0 =
      read_ndarray("./test_data/test_variable/test_exp_variable/y0.txt");
  ndarray *y1 =
      read_ndarray("./test_data/test_variable/test_exp_variable/y1.txt");
  ndarray *y2 =
      read_ndarray("./test_data/test_variable/test_exp_variable/y2.txt");
  ndarray *z0 =
      read_ndarray("./test_data/test_variable/test_exp_variable/z0.txt");
  ndarray *z1 =
      read_ndarray("./test_data/test_variable/test_exp_variable/z1.txt");
  ndarray *z2 =
      read_ndarray("./test_data/test_variable/test_exp_variable/z2.txt");
  ndarray *x0_grad =
      read_ndarray("./test_data/test_variable/test_exp_variable/x0_grad.txt");
  ndarray *x1_grad =
      read_ndarray("./test_data/test_variable/test_exp_variable/x1_grad.txt");
  ndarray *x2_grad =
      read_ndarray("./test_data/test_variable/test_exp_variable/x2_grad.txt");

  variable *var_x0 = new_variable(x0);
  variable *var_x1 = new_variable(x1);
  variable *var_x2 = new_variable(x2);

  variable *var_y0_hat = exp_variable(var_x0);
  variable *var_y1_hat = exp_variable(var_x1);
  variable *var_y2_hat = exp_variable(var_x2);
  free_ndarray(&(var_y0_hat->grad));
  free_ndarray(&(var_y1_hat->grad));
  free_ndarray(&(var_y2_hat->grad));
  var_y0_hat->grad = z0;
  var_y1_hat->grad = z1;
  var_y2_hat->grad = z2;
  backward_variable(var_y0_hat);
  backward_variable(var_y1_hat);
  backward_variable(var_y2_hat);

  assert_true(is_equal_ndarray(var_y0_hat->val, y0, NDARRAY_TYPE_EPSILON));
  assert_true(is_equal_ndarray(var_y1_hat->val, y1, NDARRAY_TYPE_EPSILON));
  assert_true(is_equal_ndarray(var_y2_hat->val, y2, NDARRAY_TYPE_EPSILON));
  assert_true(is_equal_ndarray(var_x0->grad, x0_grad, NDARRAY_TYPE_EPSILON));
  assert_true(is_equal_ndarray(var_x1->grad, x1_grad, NDARRAY_TYPE_EPSILON));
  assert_true(is_equal_ndarray(var_x2->grad, x2_grad, NDARRAY_TYPE_EPSILON));

  free_ndarray(&x0);
  free_ndarray(&x1);
  free_ndarray(&x2);
  free_ndarray(&y0);
  free_ndarray(&y1);
  free_ndarray(&y2);
  free_ndarray(&x0_grad);
  free_ndarray(&x1_grad);
  free_ndarray(&x2_grad);

  free_graph_variable(&var_y0_hat);
  free_graph_variable(&var_y1_hat);
  free_graph_variable(&var_y2_hat);
}

static void test_relu_variable(void **state) {
  (void)state;
  ndarray *x0 =
      read_ndarray("./test_data/test_variable/test_relu_variable/x0.txt");
  ndarray *x1 =
      read_ndarray("./test_data/test_variable/test_relu_variable/x1.txt");
  ndarray *x2 =
      read_ndarray("./test_data/test_variable/test_relu_variable/x2.txt");
  ndarray *y0 =
      read_ndarray("./test_data/test_variable/test_relu_variable/y0.txt");
  ndarray *y1 =
      read_ndarray("./test_data/test_variable/test_relu_variable/y1.txt");
  ndarray *y2 =
      read_ndarray("./test_data/test_variable/test_relu_variable/y2.txt");
  ndarray *z0 =
      read_ndarray("./test_data/test_variable/test_relu_variable/z0.txt");
  ndarray *z1 =
      read_ndarray("./test_data/test_variable/test_relu_variable/z1.txt");
  ndarray *z2 =
      read_ndarray("./test_data/test_variable/test_relu_variable/z2.txt");
  ndarray *x0_grad =
      read_ndarray("./test_data/test_variable/test_relu_variable/x0_grad.txt");
  ndarray *x1_grad =
      read_ndarray("./test_data/test_variable/test_relu_variable/x1_grad.txt");
  ndarray *x2_grad =
      read_ndarray("./test_data/test_variable/test_relu_variable/x2_grad.txt");

  variable *var_x0 = new_variable(x0);
  variable *var_x1 = new_variable(x1);
  variable *var_x2 = new_variable(x2);

  variable *var_y0_hat = relu_variable(var_x0);
  variable *var_y1_hat = relu_variable(var_x1);
  variable *var_y2_hat = relu_variable(var_x2);
  free_ndarray(&(var_y0_hat->grad));
  free_ndarray(&(var_y1_hat->grad));
  free_ndarray(&(var_y2_hat->grad));
  var_y0_hat->grad = z0;
  var_y1_hat->grad = z1;
  var_y2_hat->grad = z2;
  backward_variable(var_y0_hat);
  backward_variable(var_y1_hat);
  backward_variable(var_y2_hat);

  assert_true(is_equal_ndarray(var_y0_hat->val, y0, NDARRAY_TYPE_EPSILON));
  assert_true(is_equal_ndarray(var_y1_hat->val, y1, NDARRAY_TYPE_EPSILON));
  assert_true(is_equal_ndarray(var_y2_hat->val, y2, NDARRAY_TYPE_EPSILON));
  assert_true(is_equal_ndarray(var_x0->grad, x0_grad, NDARRAY_TYPE_EPSILON));
  assert_true(is_equal_ndarray(var_x1->grad, x1_grad, NDARRAY_TYPE_EPSILON));
  assert_true(is_equal_ndarray(var_x2->grad, x2_grad, NDARRAY_TYPE_EPSILON));

  free_ndarray(&x0);
  free_ndarray(&x1);
  free_ndarray(&x2);
  free_ndarray(&y0);
  free_ndarray(&y1);
  free_ndarray(&y2);
  free_ndarray(&x0_grad);
  free_ndarray(&x1_grad);
  free_ndarray(&x2_grad);

  free_graph_variable(&var_y0_hat);
  free_graph_variable(&var_y1_hat);
  free_graph_variable(&var_y2_hat);
}

static void test_sum_variable(void **state) {
  (void)state;
  ndarray *x0 =
      read_ndarray("./test_data/test_variable/test_sum_variable/x0.txt");
  ndarray *x1 =
      read_ndarray("./test_data/test_variable/test_sum_variable/x1.txt");
  ndarray *x2 =
      read_ndarray("./test_data/test_variable/test_sum_variable/x2.txt");
  ndarray *y0 =
      read_ndarray("./test_data/test_variable/test_sum_variable/y0.txt");
  ndarray *y1 =
      read_ndarray("./test_data/test_variable/test_sum_variable/y1.txt");
  ndarray *y2 =
      read_ndarray("./test_data/test_variable/test_sum_variable/y2.txt");
  ndarray *z0 =
      read_ndarray("./test_data/test_variable/test_sum_variable/z0.txt");
  ndarray *z1 =
      read_ndarray("./test_data/test_variable/test_sum_variable/z1.txt");
  ndarray *z2 =
      read_ndarray("./test_data/test_variable/test_sum_variable/z2.txt");
  ndarray *x0_grad =
      read_ndarray("./test_data/test_variable/test_sum_variable/x0_grad.txt");
  ndarray *x1_grad =
      read_ndarray("./test_data/test_variable/test_sum_variable/x1_grad.txt");
  ndarray *x2_grad =
      read_ndarray("./test_data/test_variable/test_sum_variable/x2_grad.txt");

  variable *var_x0 = new_variable(x0);
  variable *var_x1 = new_variable(x1);
  variable *var_x2 = new_variable(x2);

  variable *var_y0_hat = sum_variable(var_x0, 0);
  variable *var_y1_hat = sum_variable(var_x1, 0);
  variable *var_y2_hat = sum_variable(var_x2, 1);
  free_ndarray(&(var_y0_hat->grad));
  free_ndarray(&(var_y1_hat->grad));
  free_ndarray(&(var_y2_hat->grad));
  var_y0_hat->grad = z0;
  var_y1_hat->grad = z1;
  var_y2_hat->grad = z2;
  backward_variable(var_y0_hat);
  backward_variable(var_y1_hat);
  backward_variable(var_y2_hat);

  assert_true(is_equal_ndarray(var_y0_hat->val, y0, NDARRAY_TYPE_EPSILON));
  assert_true(is_equal_ndarray(var_y1_hat->val, y1, NDARRAY_TYPE_EPSILON));
  assert_true(is_equal_ndarray(var_y2_hat->val, y2, NDARRAY_TYPE_EPSILON));
  assert_true(is_equal_ndarray(var_x0->grad, x0_grad, NDARRAY_TYPE_EPSILON));
  assert_true(is_equal_ndarray(var_x1->grad, x1_grad, NDARRAY_TYPE_EPSILON));
  assert_true(is_equal_ndarray(var_x2->grad, x2_grad, NDARRAY_TYPE_EPSILON));

  free_ndarray(&x0);
  free_ndarray(&x1);
  free_ndarray(&x2);
  free_ndarray(&y0);
  free_ndarray(&y1);
  free_ndarray(&y2);
  free_ndarray(&x0_grad);
  free_ndarray(&x1_grad);
  free_ndarray(&x2_grad);

  free_graph_variable(&var_y0_hat);
  free_graph_variable(&var_y1_hat);
  free_graph_variable(&var_y2_hat);
}

static void test_sigmoid_variable(void **state) {
  (void)state;
  ndarray *x0 =
      read_ndarray("./test_data/test_variable/test_sigmoid_variable/x0.txt");
  ndarray *x1 =
      read_ndarray("./test_data/test_variable/test_sigmoid_variable/x1.txt");
  ndarray *x2 =
      read_ndarray("./test_data/test_variable/test_sigmoid_variable/x2.txt");
  ndarray *y0 =
      read_ndarray("./test_data/test_variable/test_sigmoid_variable/y0.txt");
  ndarray *y1 =
      read_ndarray("./test_data/test_variable/test_sigmoid_variable/y1.txt");
  ndarray *y2 =
      read_ndarray("./test_data/test_variable/test_sigmoid_variable/y2.txt");
  ndarray *z0 =
      read_ndarray("./test_data/test_variable/test_sigmoid_variable/z0.txt");
  ndarray *z1 =
      read_ndarray("./test_data/test_variable/test_sigmoid_variable/z1.txt");
  ndarray *z2 =
      read_ndarray("./test_data/test_variable/test_sigmoid_variable/z2.txt");
  ndarray *x0_grad = read_ndarray(
      "./test_data/test_variable/test_sigmoid_variable/x0_grad.txt");
  ndarray *x1_grad = read_ndarray(
      "./test_data/test_variable/test_sigmoid_variable/x1_grad.txt");
  ndarray *x2_grad = read_ndarray(
      "./test_data/test_variable/test_sigmoid_variable/x2_grad.txt");

  variable *var_x0 = new_variable(x0);
  variable *var_x1 = new_variable(x1);
  variable *var_x2 = new_variable(x2);

  variable *var_y0_hat = sigmoid_variable(var_x0);
  variable *var_y1_hat = sigmoid_variable(var_x1);
  variable *var_y2_hat = sigmoid_variable(var_x2);
  free_ndarray(&(var_y0_hat->grad));
  free_ndarray(&(var_y1_hat->grad));
  free_ndarray(&(var_y2_hat->grad));
  var_y0_hat->grad = z0;
  var_y1_hat->grad = z1;
  var_y2_hat->grad = z2;
  backward_variable(var_y0_hat);
  backward_variable(var_y1_hat);
  backward_variable(var_y2_hat);

  assert_true(is_equal_ndarray(var_y0_hat->val, y0, NDARRAY_TYPE_EPSILON));
  assert_true(is_equal_ndarray(var_y1_hat->val, y1, NDARRAY_TYPE_EPSILON));
  assert_true(is_equal_ndarray(var_y2_hat->val, y2, NDARRAY_TYPE_EPSILON));
  assert_true(is_equal_ndarray(var_x0->grad, x0_grad, NDARRAY_TYPE_EPSILON));
  assert_true(is_equal_ndarray(var_x1->grad, x1_grad, NDARRAY_TYPE_EPSILON));
  assert_true(is_equal_ndarray(var_x2->grad, x2_grad, NDARRAY_TYPE_EPSILON));

  free_ndarray(&x0);
  free_ndarray(&x1);
  free_ndarray(&x2);
  free_ndarray(&y0);
  free_ndarray(&y1);
  free_ndarray(&y2);
  free_ndarray(&x0_grad);
  free_ndarray(&x1_grad);
  free_ndarray(&x2_grad);

  free_graph_variable(&var_y0_hat);
  free_graph_variable(&var_y1_hat);
  free_graph_variable(&var_y2_hat);
}

static void test_softmax_variable(void **state) {
  (void)state;
  ndarray *x0 =
      read_ndarray("./test_data/test_variable/test_softmax_variable/x0.txt");
  ndarray *x1 =
      read_ndarray("./test_data/test_variable/test_softmax_variable/x1.txt");
  ndarray *x2 =
      read_ndarray("./test_data/test_variable/test_softmax_variable/x2.txt");
  ndarray *y0 =
      read_ndarray("./test_data/test_variable/test_softmax_variable/y0.txt");
  ndarray *y1 =
      read_ndarray("./test_data/test_variable/test_softmax_variable/y1.txt");
  ndarray *y2 =
      read_ndarray("./test_data/test_variable/test_softmax_variable/y2.txt");
  ndarray *z0 =
      read_ndarray("./test_data/test_variable/test_softmax_variable/z0.txt");
  ndarray *z1 =
      read_ndarray("./test_data/test_variable/test_softmax_variable/z1.txt");
  ndarray *z2 =
      read_ndarray("./test_data/test_variable/test_softmax_variable/z2.txt");
  ndarray *x0_grad = read_ndarray(
      "./test_data/test_variable/test_softmax_variable/x0_grad.txt");
  ndarray *x1_grad = read_ndarray(
      "./test_data/test_variable/test_softmax_variable/x1_grad.txt");
  ndarray *x2_grad = read_ndarray(
      "./test_data/test_variable/test_softmax_variable/x2_grad.txt");

  variable *var_x0 = new_variable(x0);
  variable *var_x1 = new_variable(x1);
  variable *var_x2 = new_variable(x2);

  variable *var_y0_hat = softmax_variable(var_x0, 0);
  variable *var_y1_hat = softmax_variable(var_x1, 0);
  variable *var_y2_hat = softmax_variable(var_x2, 1);
  free_ndarray(&(var_y0_hat->grad));
  free_ndarray(&(var_y1_hat->grad));
  free_ndarray(&(var_y2_hat->grad));
  var_y0_hat->grad = z0;
  var_y1_hat->grad = z1;
  var_y2_hat->grad = z2;
  backward_variable(var_y0_hat);
  backward_variable(var_y1_hat);
  backward_variable(var_y2_hat);

  assert_true(is_equal_ndarray(var_y0_hat->val, y0, NDARRAY_TYPE_EPSILON));
  assert_true(is_equal_ndarray(var_y1_hat->val, y1, NDARRAY_TYPE_EPSILON));
  assert_true(is_equal_ndarray(var_y2_hat->val, y2, NDARRAY_TYPE_EPSILON));
  assert_true(is_equal_ndarray(var_x0->grad, x0_grad, NDARRAY_TYPE_EPSILON));
  assert_true(is_equal_ndarray(var_x1->grad, x1_grad, NDARRAY_TYPE_EPSILON));
  assert_true(is_equal_ndarray(var_x2->grad, x2_grad, NDARRAY_TYPE_EPSILON));

  free_ndarray(&x0);
  free_ndarray(&x1);
  free_ndarray(&x2);
  free_ndarray(&y0);
  free_ndarray(&y1);
  free_ndarray(&y2);
  free_ndarray(&x0_grad);
  free_ndarray(&x1_grad);
  free_ndarray(&x2_grad);

  free_graph_variable(&var_y0_hat);
  free_graph_variable(&var_y1_hat);
  free_graph_variable(&var_y2_hat);
}

static void test_tanh_variable(void **state) {
  (void)state;
  ndarray *x0 =
      read_ndarray("./test_data/test_variable/test_tanh_variable/x0.txt");
  ndarray *x1 =
      read_ndarray("./test_data/test_variable/test_tanh_variable/x1.txt");
  ndarray *x2 =
      read_ndarray("./test_data/test_variable/test_tanh_variable/x2.txt");
  ndarray *y0 =
      read_ndarray("./test_data/test_variable/test_tanh_variable/y0.txt");
  ndarray *y1 =
      read_ndarray("./test_data/test_variable/test_tanh_variable/y1.txt");
  ndarray *y2 =
      read_ndarray("./test_data/test_variable/test_tanh_variable/y2.txt");
  ndarray *z0 =
      read_ndarray("./test_data/test_variable/test_tanh_variable/z0.txt");
  ndarray *z1 =
      read_ndarray("./test_data/test_variable/test_tanh_variable/z1.txt");
  ndarray *z2 =
      read_ndarray("./test_data/test_variable/test_tanh_variable/z2.txt");
  ndarray *x0_grad =
      read_ndarray("./test_data/test_variable/test_tanh_variable/x0_grad.txt");
  ndarray *x1_grad =
      read_ndarray("./test_data/test_variable/test_tanh_variable/x1_grad.txt");
  ndarray *x2_grad =
      read_ndarray("./test_data/test_variable/test_tanh_variable/x2_grad.txt");

  variable *var_x0 = new_variable(x0);
  variable *var_x1 = new_variable(x1);
  variable *var_x2 = new_variable(x2);

  variable *var_y0_hat = tanh_variable(var_x0);
  variable *var_y1_hat = tanh_variable(var_x1);
  variable *var_y2_hat = tanh_variable(var_x2);
  free_ndarray(&(var_y0_hat->grad));
  free_ndarray(&(var_y1_hat->grad));
  free_ndarray(&(var_y2_hat->grad));
  var_y0_hat->grad = z0;
  var_y1_hat->grad = z1;
  var_y2_hat->grad = z2;
  backward_variable(var_y0_hat);
  backward_variable(var_y1_hat);
  backward_variable(var_y2_hat);

  assert_true(is_equal_ndarray(var_y0_hat->val, y0, NDARRAY_TYPE_EPSILON));
  assert_true(is_equal_ndarray(var_y1_hat->val, y1, NDARRAY_TYPE_EPSILON));
  assert_true(is_equal_ndarray(var_y2_hat->val, y2, NDARRAY_TYPE_EPSILON));
  assert_true(is_equal_ndarray(var_x0->grad, x0_grad, NDARRAY_TYPE_EPSILON));
  assert_true(is_equal_ndarray(var_x1->grad, x1_grad, NDARRAY_TYPE_EPSILON));
  assert_true(is_equal_ndarray(var_x2->grad, x2_grad, NDARRAY_TYPE_EPSILON));

  free_ndarray(&x0);
  free_ndarray(&x1);
  free_ndarray(&x2);
  free_ndarray(&y0);
  free_ndarray(&y1);
  free_ndarray(&y2);
  free_ndarray(&x0_grad);
  free_ndarray(&x1_grad);
  free_ndarray(&x2_grad);

  free_graph_variable(&var_y0_hat);
  free_graph_variable(&var_y1_hat);
  free_graph_variable(&var_y2_hat);
}

static void test_matmul_variable(void **state) {
  (void)state;
  ndarray *a0 =
      read_ndarray("./test_data/test_variable/test_matmul_variable/a0.txt");
  ndarray *a1 =
      read_ndarray("./test_data/test_variable/test_matmul_variable/a1.txt");
  ndarray *a2 =
      read_ndarray("./test_data/test_variable/test_matmul_variable/a2.txt");
  ndarray *a3 =
      read_ndarray("./test_data/test_variable/test_matmul_variable/a3.txt");
  ndarray *b0 =
      read_ndarray("./test_data/test_variable/test_matmul_variable/b0.txt");
  ndarray *b1 =
      read_ndarray("./test_data/test_variable/test_matmul_variable/b1.txt");
  ndarray *b2 =
      read_ndarray("./test_data/test_variable/test_matmul_variable/b2.txt");
  ndarray *b3 =
      read_ndarray("./test_data/test_variable/test_matmul_variable/b3.txt");
  ndarray *c0 =
      read_ndarray("./test_data/test_variable/test_matmul_variable/c0.txt");
  ndarray *c1 =
      read_ndarray("./test_data/test_variable/test_matmul_variable/c1.txt");
  ndarray *c2 =
      read_ndarray("./test_data/test_variable/test_matmul_variable/c2.txt");
  ndarray *c3 =
      read_ndarray("./test_data/test_variable/test_matmul_variable/c3.txt");
  ndarray *d0 =
      read_ndarray("./test_data/test_variable/test_matmul_variable/d0.txt");
  ndarray *d1 =
      read_ndarray("./test_data/test_variable/test_matmul_variable/d1.txt");
  ndarray *d2 =
      read_ndarray("./test_data/test_variable/test_matmul_variable/d2.txt");
  ndarray *d3 =
      read_ndarray("./test_data/test_variable/test_matmul_variable/d3.txt");
  ndarray *a0_grad = read_ndarray(
      "./test_data/test_variable/test_matmul_variable/a0_grad.txt");
  ndarray *a1_grad = read_ndarray(
      "./test_data/test_variable/test_matmul_variable/a1_grad.txt");
  ndarray *a2_grad = read_ndarray(
      "./test_data/test_variable/test_matmul_variable/a2_grad.txt");
  ndarray *a3_grad = read_ndarray(
      "./test_data/test_variable/test_matmul_variable/a3_grad.txt");
  ndarray *b0_grad = read_ndarray(
      "./test_data/test_variable/test_matmul_variable/b0_grad.txt");
  ndarray *b1_grad = read_ndarray(
      "./test_data/test_variable/test_matmul_variable/b1_grad.txt");
  ndarray *b2_grad = read_ndarray(
      "./test_data/test_variable/test_matmul_variable/b2_grad.txt");
  ndarray *b3_grad = read_ndarray(
      "./test_data/test_variable/test_matmul_variable/b3_grad.txt");

  variable *var_a0 = new_variable(a0);
  variable *var_a1 = new_variable(a1);
  variable *var_a2 = new_variable(a2);
  variable *var_a3 = new_variable(a3);
  variable *var_b0 = new_variable(b0);
  variable *var_b1 = new_variable(b1);
  variable *var_b2 = new_variable(b2);
  variable *var_b3 = new_variable(b3);
  variable *var_c0_hat = matmul_variable(var_a0, var_b0);
  variable *var_c1_hat = matmul_variable(var_a1, var_b1);
  variable *var_c2_hat = matmul_variable(var_a2, var_b2);
  variable *var_c3_hat = matmul_variable(var_a3, var_b3);
  free_ndarray(&(var_c0_hat->grad));
  free_ndarray(&(var_c1_hat->grad));
  free_ndarray(&(var_c2_hat->grad));
  free_ndarray(&(var_c3_hat->grad));
  var_c0_hat->grad = d0;
  var_c1_hat->grad = d1;
  var_c2_hat->grad = d2;
  var_c3_hat->grad = d3;
  backward_variable(var_c0_hat);
  backward_variable(var_c1_hat);
  backward_variable(var_c2_hat);
  backward_variable(var_c3_hat);

  assert_true(is_equal_ndarray(var_c0_hat->val, c0, NDARRAY_TYPE_EPSILON));
  assert_true(is_equal_ndarray(var_c1_hat->val, c1, NDARRAY_TYPE_EPSILON));
  assert_true(is_equal_ndarray(var_c2_hat->val, c2, NDARRAY_TYPE_EPSILON));
  assert_true(is_equal_ndarray(var_c3_hat->val, c3, NDARRAY_TYPE_EPSILON));
  assert_true(is_equal_ndarray(var_a0->grad, a0_grad, NDARRAY_TYPE_EPSILON));
  assert_true(is_equal_ndarray(var_a1->grad, a1_grad, NDARRAY_TYPE_EPSILON));
  assert_true(is_equal_ndarray(var_a2->grad, a2_grad, NDARRAY_TYPE_EPSILON));
  assert_true(is_equal_ndarray(var_a3->grad, a3_grad, NDARRAY_TYPE_EPSILON));
  assert_true(is_equal_ndarray(var_b0->grad, b0_grad, NDARRAY_TYPE_EPSILON));
  assert_true(is_equal_ndarray(var_b1->grad, b1_grad, NDARRAY_TYPE_EPSILON));
  assert_true(is_equal_ndarray(var_b2->grad, b2_grad, NDARRAY_TYPE_EPSILON));
  assert_true(is_equal_ndarray(var_b3->grad, b3_grad, NDARRAY_TYPE_EPSILON));

  free_ndarray(&a0);
  free_ndarray(&a1);
  free_ndarray(&a2);
  free_ndarray(&a3);
  free_ndarray(&b0);
  free_ndarray(&b1);
  free_ndarray(&b2);
  free_ndarray(&b3);
  free_ndarray(&c0);
  free_ndarray(&c1);
  free_ndarray(&c2);
  free_ndarray(&c3);
  free_ndarray(&a0_grad);
  free_ndarray(&a1_grad);
  free_ndarray(&a2_grad);
  free_ndarray(&a3_grad);
  free_ndarray(&b0_grad);
  free_ndarray(&b1_grad);
  free_ndarray(&b2_grad);
  free_ndarray(&b3_grad);

  free_graph_variable(&var_c0_hat);
  free_graph_variable(&var_c1_hat);
  free_graph_variable(&var_c2_hat);
  free_graph_variable(&var_c3_hat);
}

static void test_backward_variable(void **state) {
  (void)state;
  ndarray *a =
      read_ndarray("./test_data/test_variable/test_backward_variable/a.txt");
  ndarray *b =
      read_ndarray("./test_data/test_variable/test_backward_variable/b.txt");
  ndarray *c =
      read_ndarray("./test_data/test_variable/test_backward_variable/c.txt");
  ndarray *d =
      read_ndarray("./test_data/test_variable/test_backward_variable/d.txt");
  ndarray *e =
      read_ndarray("./test_data/test_variable/test_backward_variable/e.txt");
  ndarray *a_grad = read_ndarray(
      "./test_data/test_variable/test_backward_variable/a_grad.txt");
  ndarray *b_grad = read_ndarray(
      "./test_data/test_variable/test_backward_variable/b_grad.txt");
  ndarray *c_grad = read_ndarray(
      "./test_data/test_variable/test_backward_variable/c_grad.txt");

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
  const struct CMUnitTest tests[] = {
      cmocka_unit_test(test_new_variable),
      cmocka_unit_test(test_add_variable),
      cmocka_unit_test(test_subtract_variable),
      cmocka_unit_test(test_multiply_variable),
      cmocka_unit_test(test_divide_variable),
      cmocka_unit_test(test_power_variable),
      cmocka_unit_test(test_exp_variable),
      cmocka_unit_test(test_relu_variable),
      cmocka_unit_test(test_sum_variable),
      cmocka_unit_test(test_sigmoid_variable),
      cmocka_unit_test(test_softmax_variable),
      cmocka_unit_test(test_tanh_variable),
      cmocka_unit_test(test_matmul_variable),
      cmocka_unit_test(test_backward_variable),
  };
  return cmocka_run_group_tests(tests, NULL, NULL);
}
