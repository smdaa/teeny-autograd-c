#include <math.h>
#include <setjmp.h>
#include <stdarg.h>
#include <stddef.h>
#include <stdlib.h>
#include <cmocka.h>

// valgrind --leak-check=full --show-leak-kinds=all --errors-for-leak-kinds=all --error-exitcode=1 ./test_ndarray

#include "../src/ndarray.h"

static void test_full_ndarray(void **state) {
  (void)state;

  ndarray *arr = full_ndarray(2, (int[]){2, 3}, 5.0);

  assert_non_null(arr);
  assert_int_equal(arr->dim, 2);
  assert_int_equal(arr->size, 6);
  assert_int_equal(arr->shape[0], 2);
  assert_int_equal(arr->shape[1], 3);
  for (int i = 0; i < arr->size; i++) {
    assert_float_equal(arr->data[i], 5.0, NDARRAY_TYPE_EPSILON);
  }

  free_ndarray(&arr);
}

static void test_copy_ndarray(void **state) {
  (void)state;

  ndarray *src = full_ndarray(2, (int[]){2, 3}, 5);
  ndarray *dst = copy_ndarray(src);

  assert_non_null(dst);
  assert_int_equal(dst->dim, src->dim);
  assert_int_equal(dst->size, src->size);
  assert_int_equal(dst->shape[0], src->shape[0]);
  assert_int_equal(dst->shape[1], src->shape[1]);
  for (int i = 0; i < dst->size; i++) {
    assert_float_equal(dst->data[i], src->data[i], NDARRAY_TYPE_EPSILON);
  }

  free_ndarray(&src);
  free_ndarray(&dst);
}

static void test_empty_like_ndarray(void **state) {
  (void)state;

  ndarray *src = full_ndarray(2, (int[]){2, 3}, 5);
  ndarray *dst = empty_like_ndarray(src);

  assert_non_null(dst);
  assert_int_equal(dst->dim, src->dim);
  assert_int_equal(dst->size, src->size);
  assert_int_equal(dst->shape[0], src->shape[0]);
  assert_int_equal(dst->shape[1], src->shape[1]);

  free_ndarray(&src);
  free_ndarray(&dst);
}

static void test_zeros_ndarray(void **state) {
  (void)state;

  ndarray *arr = zeros_ndarray(2, (int[]){2, 3});

  assert_non_null(arr);
  assert_int_equal(arr->dim, 2);
  assert_int_equal(arr->size, 6);
  assert_int_equal(arr->shape[0], 2);
  assert_int_equal(arr->shape[1], 3);
  for (int i = 0; i < arr->size; i++) {
    assert_float_equal(arr->data[i], 0.0, NDARRAY_TYPE_EPSILON);
  }

  free_ndarray(&arr);
}

static void test_ones_ndarray(void **state) {
  (void)state;

  ndarray *arr = ones_ndarray(2, (int[]){2, 3});

  assert_non_null(arr);
  assert_int_equal(arr->dim, 2);
  assert_int_equal(arr->size, 6);
  assert_int_equal(arr->shape[0], 2);
  assert_int_equal(arr->shape[1], 3);
  for (int i = 0; i < arr->size; i++) {
    assert_float_equal(arr->data[i], 1.0, NDARRAY_TYPE_EPSILON);
  }

  free_ndarray(&arr);
}

static void test_eye_ndarray(void **state) {
  (void)state;

  ndarray *arr = eye_ndarray(5);

  assert_non_null(arr);
  assert_int_equal(arr->dim, 2);
  assert_int_equal(arr->size, 25);
  assert_int_equal(arr->shape[0], 5);
  assert_int_equal(arr->shape[1], 5);
  for (int i = 0; i < 5; i++) {
    for (int j = 0; j < 5; j++) {
      if (i == j) {
        assert_float_equal(arr->data[i * 5 + j], 1.0, NDARRAY_TYPE_EPSILON);
      } else {
        assert_float_equal(arr->data[i * 5 + j], 0.0, NDARRAY_TYPE_EPSILON);
      }
    }
  }

  free_ndarray(&arr);
}

static void test_random_ndrray(void **state) {
  (void)state;

  ndarray *arr = random_ndrray(2, (int[]){2, 3});

  assert_non_null(arr);
  assert_int_equal(arr->dim, 2);
  assert_int_equal(arr->size, 6);
  assert_int_equal(arr->shape[0], 2);
  assert_int_equal(arr->shape[1], 3);
  for (int i = 0; i < arr->size; i++) {
    assert_in_range(arr->data[0], 0.0f, 1.0f);
  }

  free_ndarray(&arr);
}

static void test_random_truncated_ndarray(void **state) {
  (void)state;

  ndarray *arr =
      random_truncated_ndarray(2, (int[]){2, 3}, 0.0, 1.0, -0.5, 0.5);

  assert_non_null(arr);
  assert_int_equal(arr->dim, 2);
  assert_int_equal(arr->size, 6);
  assert_int_equal(arr->shape[0], 2);
  assert_int_equal(arr->shape[1], 3);
  for (int i = 0; i < arr->size; i++) {
    assert_in_range(arr->data[i], -1.0f, 1.0f);
  }

  free_ndarray(&arr);
}

static void test_read_ndarray(void **state) {
  (void)state;

  ndarray *x = read_ndarray("./test_data/test_ndarray/test_read_ndarray/x.txt");

  assert_non_null(x);
  assert_int_equal(x->dim, 2);
  assert_int_equal(x->size, 100 * 100);
  assert_int_equal(x->shape[0], 100);
  assert_int_equal(x->shape[1], 100);
  for (int i = 0; i < 100; i++) {
    for (int j = 0; j < 100; j++) {
      if (i == j) {
        assert_float_equal(x->data[i * 100 + j], 2.0, NDARRAY_TYPE_EPSILON);
      } else {
        assert_float_equal(x->data[i * 100 + j], 1.0, NDARRAY_TYPE_EPSILON);
      }
    }
  }

  free_ndarray(&x);
}

static void test_is_equal(void **state) {
  (void)state;

  ndarray arr1 = {2, 4, (int[]){2, 2}, (NDARRAY_TYPE[]){1, 2, 3, 4}};
  ndarray arr2 = {2, 4, (int[]){2, 2}, (NDARRAY_TYPE[]){1, 2, 3, 4}};
  ndarray arr3 = {2, 4, (int[]){2, 2}, (NDARRAY_TYPE[]){1, 2, 3, 4}};
  ndarray arr4 = {3, 4, (int[]){2, 2, 1}, (NDARRAY_TYPE[]){1, 2, 3, 4}};
  ndarray arr5 = {2, 4, (int[]){2, 2}, (NDARRAY_TYPE[]){1, 2, 3, 4}};
  ndarray arr6 = {2, 4, (int[]){2, 2}, (NDARRAY_TYPE[]){1, 2, 5, 4}};

  assert_true(is_equal_ndarray(&arr1, &arr2, NDARRAY_TYPE_EPSILON));
  assert_false(is_equal_ndarray(&arr3, &arr4, NDARRAY_TYPE_EPSILON));
  assert_false(is_equal_ndarray(&arr5, &arr6, NDARRAY_TYPE_EPSILON));
}

static void test_unary_op_ndarray(void **state) {
  (void)state;

  ndarray *x =
      read_ndarray("./test_data/test_ndarray/test_unary_op_ndarray/x.txt");
  ndarray *y =
      read_ndarray("./test_data/test_ndarray/test_unary_op_ndarray/y.txt");
  ndarray *y_hat = unary_op_ndarray(x, sin);

  assert_true(is_equal_ndarray(y, y_hat, NDARRAY_TYPE_EPSILON));

  free_ndarray(&x);
  free_ndarray(&y);
  free_ndarray(&y_hat);
}

static void test_log_ndarray(void **state) {
  (void)state;

  ndarray *x = read_ndarray("./test_data/test_ndarray/test_log_ndarray/x.txt");
  ndarray *y = read_ndarray("./test_data/test_ndarray/test_log_ndarray/y.txt");
  ndarray *y_hat = log_ndarray(x);

  assert_true(is_equal_ndarray(y, y_hat, NDARRAY_TYPE_EPSILON));

  free_ndarray(&x);
  free_ndarray(&y);
  free_ndarray(&y_hat);
}

static void test_add_ndarray_scalar(void **state) {
  (void)state;

  ndarray *x =
      read_ndarray("./test_data/test_ndarray/test_add_ndarray_scalar/x.txt");
  ndarray *y =
      read_ndarray("./test_data/test_ndarray/test_add_ndarray_scalar/y.txt");
  ndarray *y_hat = add_ndarray_scalar(x, 10);

  assert_true(is_equal_ndarray(y, y_hat, NDARRAY_TYPE_EPSILON));

  free_ndarray(&x);
  free_ndarray(&y);
  free_ndarray(&y_hat);
}

static void test_subtract_ndarray_scalar(void **state) {
  (void)state;

  ndarray *x = read_ndarray(
      "./test_data/test_ndarray/test_subtract_ndarray_scalar/x.txt");
  ndarray *y = read_ndarray(
      "./test_data/test_ndarray/test_subtract_ndarray_scalar/y.txt");
  ndarray *y_hat = subtract_ndarray_scalar(x, 10);

  assert_true(is_equal_ndarray(y, y_hat, NDARRAY_TYPE_EPSILON));

  free_ndarray(&x);
  free_ndarray(&y);
  free_ndarray(&y_hat);
}

static void test_multiply_ndarray_scalar(void **state) {
  (void)state;

  ndarray *x = read_ndarray(
      "./test_data/test_ndarray/test_multiply_ndarray_scalar/x.txt");
  ndarray *y = read_ndarray(
      "./test_data/test_ndarray/test_multiply_ndarray_scalar/y.txt");
  ndarray *y_hat = multiply_ndarray_scalar(x, 10);

  assert_true(is_equal_ndarray(y, y_hat, NDARRAY_TYPE_EPSILON));

  free_ndarray(&x);
  free_ndarray(&y);
  free_ndarray(&y_hat);
}

static void test_divide_ndarray_scalar(void **state) {
  (void)state;

  ndarray *x =
      read_ndarray("./test_data/test_ndarray/test_divide_ndarray_scalar/x.txt");
  ndarray *y =
      read_ndarray("./test_data/test_ndarray/test_divide_ndarray_scalar/y.txt");
  ndarray *y_hat = divide_ndarray_scalar(x, 10);

  assert_true(is_equal_ndarray(y, y_hat, NDARRAY_TYPE_EPSILON));

  free_ndarray(&x);
  free_ndarray(&y);
  free_ndarray(&y_hat);
}

static void test_divide_scalar_ndarray(void **state) {
  (void)state;

  ndarray *x =
      read_ndarray("./test_data/test_ndarray/test_divide_scalar_ndarray/x.txt");
  ndarray *y =
      read_ndarray("./test_data/test_ndarray/test_divide_scalar_ndarray/y.txt");
  ndarray *y_hat = divide_scalar_ndarray(x, 10);

  assert_true(is_equal_ndarray(y, y_hat, NDARRAY_TYPE_EPSILON));

  free_ndarray(&x);
  free_ndarray(&y);
  free_ndarray(&y_hat);
}

static void test_power_ndarray_scalar(void **state) {
  (void)state;

  ndarray *x =
      read_ndarray("./test_data/test_ndarray/test_power_ndarray_scalar/x.txt");
  ndarray *y =
      read_ndarray("./test_data/test_ndarray/test_power_ndarray_scalar/y.txt");
  ndarray *y_hat = power_ndarray_scalar(x, 2);

  assert_true(is_equal_ndarray(y, y_hat, NDARRAY_TYPE_EPSILON));

  free_ndarray(&x);
  free_ndarray(&y);
  free_ndarray(&y_hat);
}

static void test_binary_op_ndarray(void **state) {
  (void)state;

  ndarray *a0 =
      read_ndarray("./test_data/test_ndarray/test_binary_op_ndarray/a0.txt");
  ndarray *a1 =
      read_ndarray("./test_data/test_ndarray/test_binary_op_ndarray/a1.txt");
  ndarray *a2 =
      read_ndarray("./test_data/test_ndarray/test_binary_op_ndarray/a2.txt");
  ndarray *a3 =
      read_ndarray("./test_data/test_ndarray/test_binary_op_ndarray/a3.txt");
  ndarray *a4 =
      read_ndarray("./test_data/test_ndarray/test_binary_op_ndarray/a4.txt");
  ndarray *a5 =
      read_ndarray("./test_data/test_ndarray/test_binary_op_ndarray/a5.txt");
  ndarray *b0 =
      read_ndarray("./test_data/test_ndarray/test_binary_op_ndarray/b0.txt");
  ndarray *b1 =
      read_ndarray("./test_data/test_ndarray/test_binary_op_ndarray/b1.txt");
  ndarray *b2 =
      read_ndarray("./test_data/test_ndarray/test_binary_op_ndarray/b2.txt");
  ndarray *b3 =
      read_ndarray("./test_data/test_ndarray/test_binary_op_ndarray/b3.txt");
  ndarray *b4 =
      read_ndarray("./test_data/test_ndarray/test_binary_op_ndarray/b4.txt");
  ndarray *b5 =
      read_ndarray("./test_data/test_ndarray/test_binary_op_ndarray/b5.txt");
  ndarray *c0 =
      read_ndarray("./test_data/test_ndarray/test_binary_op_ndarray/c0.txt");
  ndarray *c1 =
      read_ndarray("./test_data/test_ndarray/test_binary_op_ndarray/c1.txt");
  ndarray *c2 =
      read_ndarray("./test_data/test_ndarray/test_binary_op_ndarray/c2.txt");
  ndarray *c3 =
      read_ndarray("./test_data/test_ndarray/test_binary_op_ndarray/c3.txt");
  ndarray *c4 =
      read_ndarray("./test_data/test_ndarray/test_binary_op_ndarray/c4.txt");
  ndarray *c5 =
      read_ndarray("./test_data/test_ndarray/test_binary_op_ndarray/c5.txt");
  ndarray *c0_hat = binary_op_ndarray(a0, b0, fmax);
  ndarray *c1_hat = binary_op_ndarray(a1, b1, fmax);
  ndarray *c2_hat = binary_op_ndarray(a2, b2, fmax);
  ndarray *c3_hat = binary_op_ndarray(a3, b3, fmax);
  ndarray *c4_hat = binary_op_ndarray(a4, b4, fmax);
  ndarray *c5_hat = binary_op_ndarray(a5, b5, fmax);

  assert_true(is_equal_ndarray(c0, c0_hat, NDARRAY_TYPE_EPSILON));
  assert_true(is_equal_ndarray(c1, c1_hat, NDARRAY_TYPE_EPSILON));
  assert_true(is_equal_ndarray(c2, c2_hat, NDARRAY_TYPE_EPSILON));
  assert_true(is_equal_ndarray(c3, c3_hat, NDARRAY_TYPE_EPSILON));
  assert_true(is_equal_ndarray(c4, c4_hat, NDARRAY_TYPE_EPSILON));
  assert_true(is_equal_ndarray(c5, c5_hat, NDARRAY_TYPE_EPSILON));

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
  free_ndarray(&c0_hat);
  free_ndarray(&c1_hat);
  free_ndarray(&c2_hat);
  free_ndarray(&c3_hat);
  free_ndarray(&c4_hat);
  free_ndarray(&c5_hat);
}

static void test_add_ndarray_ndarray(void **state) {
  (void)state;

  ndarray *a0 =
      read_ndarray("./test_data/test_ndarray/test_add_ndarray_ndarray/a0.txt");
  ndarray *a1 =
      read_ndarray("./test_data/test_ndarray/test_add_ndarray_ndarray/a1.txt");
  ndarray *a2 =
      read_ndarray("./test_data/test_ndarray/test_add_ndarray_ndarray/a2.txt");
  ndarray *a3 =
      read_ndarray("./test_data/test_ndarray/test_add_ndarray_ndarray/a3.txt");
  ndarray *a4 =
      read_ndarray("./test_data/test_ndarray/test_add_ndarray_ndarray/a4.txt");
  ndarray *a5 =
      read_ndarray("./test_data/test_ndarray/test_add_ndarray_ndarray/a5.txt");
  ndarray *b0 =
      read_ndarray("./test_data/test_ndarray/test_add_ndarray_ndarray/b0.txt");
  ndarray *b1 =
      read_ndarray("./test_data/test_ndarray/test_add_ndarray_ndarray/b1.txt");
  ndarray *b2 =
      read_ndarray("./test_data/test_ndarray/test_add_ndarray_ndarray/b2.txt");
  ndarray *b3 =
      read_ndarray("./test_data/test_ndarray/test_add_ndarray_ndarray/b3.txt");
  ndarray *b4 =
      read_ndarray("./test_data/test_ndarray/test_add_ndarray_ndarray/b4.txt");
  ndarray *b5 =
      read_ndarray("./test_data/test_ndarray/test_add_ndarray_ndarray/b5.txt");
  ndarray *c0 =
      read_ndarray("./test_data/test_ndarray/test_add_ndarray_ndarray/c0.txt");
  ndarray *c1 =
      read_ndarray("./test_data/test_ndarray/test_add_ndarray_ndarray/c1.txt");
  ndarray *c2 =
      read_ndarray("./test_data/test_ndarray/test_add_ndarray_ndarray/c2.txt");
  ndarray *c3 =
      read_ndarray("./test_data/test_ndarray/test_add_ndarray_ndarray/c3.txt");
  ndarray *c4 =
      read_ndarray("./test_data/test_ndarray/test_add_ndarray_ndarray/c4.txt");
  ndarray *c5 =
      read_ndarray("./test_data/test_ndarray/test_add_ndarray_ndarray/c5.txt");
  ndarray *c0_hat = add_ndarray_ndarray(a0, b0);
  ndarray *c1_hat = add_ndarray_ndarray(a1, b1);
  ndarray *c2_hat = add_ndarray_ndarray(a2, b2);
  ndarray *c3_hat = add_ndarray_ndarray(a3, b3);
  ndarray *c4_hat = add_ndarray_ndarray(a4, b4);
  ndarray *c5_hat = add_ndarray_ndarray(a5, b5);

  assert_true(is_equal_ndarray(c0, c0_hat, NDARRAY_TYPE_EPSILON));
  assert_true(is_equal_ndarray(c1, c1_hat, NDARRAY_TYPE_EPSILON));
  assert_true(is_equal_ndarray(c2, c2_hat, NDARRAY_TYPE_EPSILON));
  assert_true(is_equal_ndarray(c3, c3_hat, NDARRAY_TYPE_EPSILON));
  assert_true(is_equal_ndarray(c4, c4_hat, NDARRAY_TYPE_EPSILON));
  assert_true(is_equal_ndarray(c5, c5_hat, NDARRAY_TYPE_EPSILON));

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
  free_ndarray(&c0_hat);
  free_ndarray(&c1_hat);
  free_ndarray(&c2_hat);
  free_ndarray(&c3_hat);
  free_ndarray(&c4_hat);
  free_ndarray(&c5_hat);
}

static void test_subtract_ndarray_ndarray(void **state) {
  (void)state;

  ndarray *a0 = read_ndarray(
      "./test_data/test_ndarray/test_subtract_ndarray_ndarray/a0.txt");
  ndarray *a1 = read_ndarray(
      "./test_data/test_ndarray/test_subtract_ndarray_ndarray/a1.txt");
  ndarray *a2 = read_ndarray(
      "./test_data/test_ndarray/test_subtract_ndarray_ndarray/a2.txt");
  ndarray *a3 = read_ndarray(
      "./test_data/test_ndarray/test_subtract_ndarray_ndarray/a3.txt");
  ndarray *a4 = read_ndarray(
      "./test_data/test_ndarray/test_subtract_ndarray_ndarray/a4.txt");
  ndarray *a5 = read_ndarray(
      "./test_data/test_ndarray/test_subtract_ndarray_ndarray/a5.txt");
  ndarray *b0 = read_ndarray(
      "./test_data/test_ndarray/test_subtract_ndarray_ndarray/b0.txt");
  ndarray *b1 = read_ndarray(
      "./test_data/test_ndarray/test_subtract_ndarray_ndarray/b1.txt");
  ndarray *b2 = read_ndarray(
      "./test_data/test_ndarray/test_subtract_ndarray_ndarray/b2.txt");
  ndarray *b3 = read_ndarray(
      "./test_data/test_ndarray/test_subtract_ndarray_ndarray/b3.txt");
  ndarray *b4 = read_ndarray(
      "./test_data/test_ndarray/test_subtract_ndarray_ndarray/b4.txt");
  ndarray *b5 = read_ndarray(
      "./test_data/test_ndarray/test_subtract_ndarray_ndarray/b5.txt");
  ndarray *c0 = read_ndarray(
      "./test_data/test_ndarray/test_subtract_ndarray_ndarray/c0.txt");
  ndarray *c1 = read_ndarray(
      "./test_data/test_ndarray/test_subtract_ndarray_ndarray/c1.txt");
  ndarray *c2 = read_ndarray(
      "./test_data/test_ndarray/test_subtract_ndarray_ndarray/c2.txt");
  ndarray *c3 = read_ndarray(
      "./test_data/test_ndarray/test_subtract_ndarray_ndarray/c3.txt");
  ndarray *c4 = read_ndarray(
      "./test_data/test_ndarray/test_subtract_ndarray_ndarray/c4.txt");
  ndarray *c5 = read_ndarray(
      "./test_data/test_ndarray/test_subtract_ndarray_ndarray/c5.txt");
  ndarray *c0_hat = subtract_ndarray_ndarray(a0, b0);
  ndarray *c1_hat = subtract_ndarray_ndarray(a1, b1);
  ndarray *c2_hat = subtract_ndarray_ndarray(a2, b2);
  ndarray *c3_hat = subtract_ndarray_ndarray(a3, b3);
  ndarray *c4_hat = subtract_ndarray_ndarray(a4, b4);
  ndarray *c5_hat = subtract_ndarray_ndarray(a5, b5);

  assert_true(is_equal_ndarray(c0, c0_hat, NDARRAY_TYPE_EPSILON));
  assert_true(is_equal_ndarray(c1, c1_hat, NDARRAY_TYPE_EPSILON));
  assert_true(is_equal_ndarray(c2, c2_hat, NDARRAY_TYPE_EPSILON));
  assert_true(is_equal_ndarray(c3, c3_hat, NDARRAY_TYPE_EPSILON));
  assert_true(is_equal_ndarray(c4, c4_hat, NDARRAY_TYPE_EPSILON));
  assert_true(is_equal_ndarray(c5, c5_hat, NDARRAY_TYPE_EPSILON));

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
  free_ndarray(&c0_hat);
  free_ndarray(&c1_hat);
  free_ndarray(&c2_hat);
  free_ndarray(&c3_hat);
  free_ndarray(&c4_hat);
  free_ndarray(&c5_hat);
}

static void test_multiply_ndarray_ndarray(void **state) {
  (void)state;

  ndarray *a0 = read_ndarray(
      "./test_data/test_ndarray/test_multiply_ndarray_ndarray/a0.txt");
  ndarray *a1 = read_ndarray(
      "./test_data/test_ndarray/test_multiply_ndarray_ndarray/a1.txt");
  ndarray *a2 = read_ndarray(
      "./test_data/test_ndarray/test_multiply_ndarray_ndarray/a2.txt");
  ndarray *a3 = read_ndarray(
      "./test_data/test_ndarray/test_multiply_ndarray_ndarray/a3.txt");
  ndarray *a4 = read_ndarray(
      "./test_data/test_ndarray/test_multiply_ndarray_ndarray/a4.txt");
  ndarray *a5 = read_ndarray(
      "./test_data/test_ndarray/test_multiply_ndarray_ndarray/a5.txt");
  ndarray *b0 = read_ndarray(
      "./test_data/test_ndarray/test_multiply_ndarray_ndarray/b0.txt");
  ndarray *b1 = read_ndarray(
      "./test_data/test_ndarray/test_multiply_ndarray_ndarray/b1.txt");
  ndarray *b2 = read_ndarray(
      "./test_data/test_ndarray/test_multiply_ndarray_ndarray/b2.txt");
  ndarray *b3 = read_ndarray(
      "./test_data/test_ndarray/test_multiply_ndarray_ndarray/b3.txt");
  ndarray *b4 = read_ndarray(
      "./test_data/test_ndarray/test_multiply_ndarray_ndarray/b4.txt");
  ndarray *b5 = read_ndarray(
      "./test_data/test_ndarray/test_multiply_ndarray_ndarray/b5.txt");
  ndarray *c0 = read_ndarray(
      "./test_data/test_ndarray/test_multiply_ndarray_ndarray/c0.txt");
  ndarray *c1 = read_ndarray(
      "./test_data/test_ndarray/test_multiply_ndarray_ndarray/c1.txt");
  ndarray *c2 = read_ndarray(
      "./test_data/test_ndarray/test_multiply_ndarray_ndarray/c2.txt");
  ndarray *c3 = read_ndarray(
      "./test_data/test_ndarray/test_multiply_ndarray_ndarray/c3.txt");
  ndarray *c4 = read_ndarray(
      "./test_data/test_ndarray/test_multiply_ndarray_ndarray/c4.txt");
  ndarray *c5 = read_ndarray(
      "./test_data/test_ndarray/test_multiply_ndarray_ndarray/c5.txt");
  ndarray *c0_hat = multiply_ndarray_ndarray(a0, b0);
  ndarray *c1_hat = multiply_ndarray_ndarray(a1, b1);
  ndarray *c2_hat = multiply_ndarray_ndarray(a2, b2);
  ndarray *c3_hat = multiply_ndarray_ndarray(a3, b3);
  ndarray *c4_hat = multiply_ndarray_ndarray(a4, b4);
  ndarray *c5_hat = multiply_ndarray_ndarray(a5, b5);

  assert_true(is_equal_ndarray(c0, c0_hat, NDARRAY_TYPE_EPSILON));
  assert_true(is_equal_ndarray(c1, c1_hat, NDARRAY_TYPE_EPSILON));
  assert_true(is_equal_ndarray(c2, c2_hat, NDARRAY_TYPE_EPSILON));
  assert_true(is_equal_ndarray(c3, c3_hat, NDARRAY_TYPE_EPSILON));
  assert_true(is_equal_ndarray(c4, c4_hat, NDARRAY_TYPE_EPSILON));
  assert_true(is_equal_ndarray(c5, c5_hat, NDARRAY_TYPE_EPSILON));

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
  free_ndarray(&c0_hat);
  free_ndarray(&c1_hat);
  free_ndarray(&c2_hat);
  free_ndarray(&c3_hat);
  free_ndarray(&c4_hat);
  free_ndarray(&c5_hat);
}

static void test_divide_ndarray_ndarray(void **state) {
  (void)state;

  ndarray *a0 = read_ndarray(
      "./test_data/test_ndarray/test_divide_ndarray_ndarray/a0.txt");
  ndarray *a1 = read_ndarray(
      "./test_data/test_ndarray/test_divide_ndarray_ndarray/a1.txt");
  ndarray *a2 = read_ndarray(
      "./test_data/test_ndarray/test_divide_ndarray_ndarray/a2.txt");
  ndarray *a3 = read_ndarray(
      "./test_data/test_ndarray/test_divide_ndarray_ndarray/a3.txt");
  ndarray *a4 = read_ndarray(
      "./test_data/test_ndarray/test_divide_ndarray_ndarray/a4.txt");
  ndarray *a5 = read_ndarray(
      "./test_data/test_ndarray/test_divide_ndarray_ndarray/a5.txt");
  ndarray *b0 = read_ndarray(
      "./test_data/test_ndarray/test_divide_ndarray_ndarray/b0.txt");
  ndarray *b1 = read_ndarray(
      "./test_data/test_ndarray/test_divide_ndarray_ndarray/b1.txt");
  ndarray *b2 = read_ndarray(
      "./test_data/test_ndarray/test_divide_ndarray_ndarray/b2.txt");
  ndarray *b3 = read_ndarray(
      "./test_data/test_ndarray/test_divide_ndarray_ndarray/b3.txt");
  ndarray *b4 = read_ndarray(
      "./test_data/test_ndarray/test_divide_ndarray_ndarray/b4.txt");
  ndarray *b5 = read_ndarray(
      "./test_data/test_ndarray/test_divide_ndarray_ndarray/b5.txt");
  ndarray *c0 = read_ndarray(
      "./test_data/test_ndarray/test_divide_ndarray_ndarray/c0.txt");
  ndarray *c1 = read_ndarray(
      "./test_data/test_ndarray/test_divide_ndarray_ndarray/c1.txt");
  ndarray *c2 = read_ndarray(
      "./test_data/test_ndarray/test_divide_ndarray_ndarray/c2.txt");
  ndarray *c3 = read_ndarray(
      "./test_data/test_ndarray/test_divide_ndarray_ndarray/c3.txt");
  ndarray *c4 = read_ndarray(
      "./test_data/test_ndarray/test_divide_ndarray_ndarray/c4.txt");
  ndarray *c5 = read_ndarray(
      "./test_data/test_ndarray/test_divide_ndarray_ndarray/c5.txt");
  ndarray *c0_hat = divide_ndarray_ndarray(a0, b0);
  ndarray *c1_hat = divide_ndarray_ndarray(a1, b1);
  ndarray *c2_hat = divide_ndarray_ndarray(a2, b2);
  ndarray *c3_hat = divide_ndarray_ndarray(a3, b3);
  ndarray *c4_hat = divide_ndarray_ndarray(a4, b4);
  ndarray *c5_hat = divide_ndarray_ndarray(a5, b5);

  assert_true(is_equal_ndarray(c0, c0_hat, NDARRAY_TYPE_EPSILON));
  assert_true(is_equal_ndarray(c1, c1_hat, NDARRAY_TYPE_EPSILON));
  assert_true(is_equal_ndarray(c2, c2_hat, NDARRAY_TYPE_EPSILON));
  assert_true(is_equal_ndarray(c3, c3_hat, NDARRAY_TYPE_EPSILON));
  assert_true(is_equal_ndarray(c4, c4_hat, NDARRAY_TYPE_EPSILON));
  assert_true(is_equal_ndarray(c5, c5_hat, NDARRAY_TYPE_EPSILON));

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
  free_ndarray(&c0_hat);
  free_ndarray(&c1_hat);
  free_ndarray(&c2_hat);
  free_ndarray(&c3_hat);
  free_ndarray(&c4_hat);
  free_ndarray(&c5_hat);
}

static void test_power_ndarray_ndarray(void **state) {
  (void)state;

  ndarray *a0 = read_ndarray(
      "./test_data/test_ndarray/test_power_ndarray_ndarray/a0.txt");
  ndarray *a1 = read_ndarray(
      "./test_data/test_ndarray/test_power_ndarray_ndarray/a1.txt");
  ndarray *a2 = read_ndarray(
      "./test_data/test_ndarray/test_power_ndarray_ndarray/a2.txt");
  ndarray *a3 = read_ndarray(
      "./test_data/test_ndarray/test_power_ndarray_ndarray/a3.txt");
  ndarray *a4 = read_ndarray(
      "./test_data/test_ndarray/test_power_ndarray_ndarray/a4.txt");
  ndarray *a5 = read_ndarray(
      "./test_data/test_ndarray/test_power_ndarray_ndarray/a5.txt");
  ndarray *b0 = read_ndarray(
      "./test_data/test_ndarray/test_power_ndarray_ndarray/b0.txt");
  ndarray *b1 = read_ndarray(
      "./test_data/test_ndarray/test_power_ndarray_ndarray/b1.txt");
  ndarray *b2 = read_ndarray(
      "./test_data/test_ndarray/test_power_ndarray_ndarray/b2.txt");
  ndarray *b3 = read_ndarray(
      "./test_data/test_ndarray/test_power_ndarray_ndarray/b3.txt");
  ndarray *b4 = read_ndarray(
      "./test_data/test_ndarray/test_power_ndarray_ndarray/b4.txt");
  ndarray *b5 = read_ndarray(
      "./test_data/test_ndarray/test_power_ndarray_ndarray/b5.txt");
  ndarray *c0 = read_ndarray(
      "./test_data/test_ndarray/test_power_ndarray_ndarray/c0.txt");
  ndarray *c1 = read_ndarray(
      "./test_data/test_ndarray/test_power_ndarray_ndarray/c1.txt");
  ndarray *c2 = read_ndarray(
      "./test_data/test_ndarray/test_power_ndarray_ndarray/c2.txt");
  ndarray *c3 = read_ndarray(
      "./test_data/test_ndarray/test_power_ndarray_ndarray/c3.txt");
  ndarray *c4 = read_ndarray(
      "./test_data/test_ndarray/test_power_ndarray_ndarray/c4.txt");
  ndarray *c5 = read_ndarray(
      "./test_data/test_ndarray/test_power_ndarray_ndarray/c5.txt");
  ndarray *c0_hat = power_ndarray_ndarray(a0, b0);
  ndarray *c1_hat = power_ndarray_ndarray(a1, b1);
  ndarray *c2_hat = power_ndarray_ndarray(a2, b2);
  ndarray *c3_hat = power_ndarray_ndarray(a3, b3);
  ndarray *c4_hat = power_ndarray_ndarray(a4, b4);
  ndarray *c5_hat = power_ndarray_ndarray(a5, b5);

  assert_true(is_equal_ndarray(c0, c0_hat, NDARRAY_TYPE_EPSILON));
  assert_true(is_equal_ndarray(c1, c1_hat, NDARRAY_TYPE_EPSILON));
  assert_true(is_equal_ndarray(c2, c2_hat, NDARRAY_TYPE_EPSILON));
  assert_true(is_equal_ndarray(c3, c3_hat, NDARRAY_TYPE_EPSILON));
  assert_true(is_equal_ndarray(c4, c4_hat, NDARRAY_TYPE_EPSILON));
  assert_true(is_equal_ndarray(c5, c5_hat, NDARRAY_TYPE_EPSILON));

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
  free_ndarray(&c0_hat);
  free_ndarray(&c1_hat);
  free_ndarray(&c2_hat);
  free_ndarray(&c3_hat);
  free_ndarray(&c4_hat);
  free_ndarray(&c5_hat);
}

static void test_matmul_ndarray(void **state) {
  (void)state;

  ndarray *a0 =
      read_ndarray("./test_data/test_ndarray/test_matmul_ndarray/a0.txt");
  ndarray *a1 =
      read_ndarray("./test_data/test_ndarray/test_matmul_ndarray/a1.txt");
  ndarray *a2 =
      read_ndarray("./test_data/test_ndarray/test_matmul_ndarray/a2.txt");
  ndarray *a3 =
      read_ndarray("./test_data/test_ndarray/test_matmul_ndarray/a3.txt");
  ndarray *b0 =
      read_ndarray("./test_data/test_ndarray/test_matmul_ndarray/b0.txt");
  ndarray *b1 =
      read_ndarray("./test_data/test_ndarray/test_matmul_ndarray/b1.txt");
  ndarray *b2 =
      read_ndarray("./test_data/test_ndarray/test_matmul_ndarray/b2.txt");
  ndarray *b3 =
      read_ndarray("./test_data/test_ndarray/test_matmul_ndarray/b3.txt");
  ndarray *c0 =
      read_ndarray("./test_data/test_ndarray/test_matmul_ndarray/c0.txt");
  ndarray *c1 =
      read_ndarray("./test_data/test_ndarray/test_matmul_ndarray/c1.txt");
  ndarray *c2 =
      read_ndarray("./test_data/test_ndarray/test_matmul_ndarray/c2.txt");
  ndarray *c3 =
      read_ndarray("./test_data/test_ndarray/test_matmul_ndarray/c3.txt");
  ndarray *c0_hat = matmul_ndarray(a0, b0);
  ndarray *c1_hat = matmul_ndarray(a1, b1);
  ndarray *c2_hat = matmul_ndarray(a2, b2);
  ndarray *c3_hat = matmul_ndarray(a3, b3);

  assert_true(is_equal_ndarray(c0, c0_hat, NDARRAY_TYPE_EPSILON));
  assert_true(is_equal_ndarray(c1, c1_hat, NDARRAY_TYPE_EPSILON));
  assert_true(is_equal_ndarray(c2, c2_hat, NDARRAY_TYPE_EPSILON));
  assert_true(is_equal_ndarray(c3, c3_hat, NDARRAY_TYPE_EPSILON));

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
  free_ndarray(&c0_hat);
  free_ndarray(&c1_hat);
  free_ndarray(&c2_hat);
  free_ndarray(&c3_hat);
}

static void test_transpose_ndarray(void **state) {
  (void)state;

  ndarray *x =
      read_ndarray("./test_data/test_ndarray/test_transpose_ndarray/x.txt");
  ndarray *y0 =
      read_ndarray("./test_data/test_ndarray/test_transpose_ndarray/y0.txt");
  ndarray *y1 =
      read_ndarray("./test_data/test_ndarray/test_transpose_ndarray/y1.txt");
  ndarray *y2 =
      read_ndarray("./test_data/test_ndarray/test_transpose_ndarray/y2.txt");
  ndarray *y0_hat = transpose_ndarray(x, (int[]){1, 0, 2});
  ndarray *y1_hat = transpose_ndarray(x, (int[]){2, 1, 0});
  ndarray *y2_hat = transpose_ndarray(x, (int[]){0, 2, 1});

  assert_true(is_equal_ndarray(y0, y0_hat, NDARRAY_TYPE_EPSILON));
  assert_true(is_equal_ndarray(y1, y1_hat, NDARRAY_TYPE_EPSILON));
  assert_true(is_equal_ndarray(y2, y2_hat, NDARRAY_TYPE_EPSILON));

  free_ndarray(&x);
  free_ndarray(&y0);
  free_ndarray(&y1);
  free_ndarray(&y2);
  free_ndarray(&y0_hat);
  free_ndarray(&y1_hat);
  free_ndarray(&y2_hat);
}

static void test_max_ndarray(void **state) {
  (void)state;

  ndarray *x = read_ndarray("./test_data/test_ndarray/test_max_ndarray/x.txt");
  ndarray *y0 =
      read_ndarray("./test_data/test_ndarray/test_max_ndarray/y0.txt");
  ndarray *y1 =
      read_ndarray("./test_data/test_ndarray/test_max_ndarray/y1.txt");
  ndarray *y2 =
      read_ndarray("./test_data/test_ndarray/test_max_ndarray/y2.txt");
  ndarray *y0_hat = max_ndarray(x, 0);
  ndarray *y1_hat = max_ndarray(x, 1);
  ndarray *y2_hat = max_ndarray(x, 2);

  assert_true(is_equal_ndarray(y0, y0_hat, NDARRAY_TYPE_EPSILON));
  assert_true(is_equal_ndarray(y1, y1_hat, NDARRAY_TYPE_EPSILON));
  assert_true(is_equal_ndarray(y2, y2_hat, NDARRAY_TYPE_EPSILON));

  free_ndarray(&x);
  free_ndarray(&y0);
  free_ndarray(&y1);
  free_ndarray(&y2);
  free_ndarray(&y0_hat);
  free_ndarray(&y1_hat);
  free_ndarray(&y2_hat);
}

static void test_min_ndarray(void **state) {
  (void)state;

  ndarray *x = read_ndarray("./test_data/test_ndarray/test_min_ndarray/x.txt");
  ndarray *y0 =
      read_ndarray("./test_data/test_ndarray/test_min_ndarray/y0.txt");
  ndarray *y1 =
      read_ndarray("./test_data/test_ndarray/test_min_ndarray/y1.txt");
  ndarray *y2 =
      read_ndarray("./test_data/test_ndarray/test_min_ndarray/y2.txt");
  ndarray *y0_hat = min_ndarray(x, 0);
  ndarray *y1_hat = min_ndarray(x, 1);
  ndarray *y2_hat = min_ndarray(x, 2);

  assert_true(is_equal_ndarray(y0, y0_hat, NDARRAY_TYPE_EPSILON));
  assert_true(is_equal_ndarray(y1, y1_hat, NDARRAY_TYPE_EPSILON));
  assert_true(is_equal_ndarray(y2, y2_hat, NDARRAY_TYPE_EPSILON));

  free_ndarray(&x);
  free_ndarray(&y0);
  free_ndarray(&y1);
  free_ndarray(&y2);
  free_ndarray(&y0_hat);
  free_ndarray(&y1_hat);
  free_ndarray(&y2_hat);
}

static void test_sum_ndarray(void **state) {
  (void)state;

  ndarray *x = read_ndarray("./test_data/test_ndarray/test_sum_ndarray/x.txt");
  ndarray *y0 =
      read_ndarray("./test_data/test_ndarray/test_sum_ndarray/y0.txt");
  ndarray *y1 =
      read_ndarray("./test_data/test_ndarray/test_sum_ndarray/y1.txt");
  ndarray *y2 =
      read_ndarray("./test_data/test_ndarray/test_sum_ndarray/y2.txt");
  ndarray *y0_hat = sum_ndarray(x, 0);
  ndarray *y1_hat = sum_ndarray(x, 1);
  ndarray *y2_hat = sum_ndarray(x, 2);

  assert_true(is_equal_ndarray(y0, y0_hat, NDARRAY_TYPE_EPSILON));
  assert_true(is_equal_ndarray(y1, y1_hat, NDARRAY_TYPE_EPSILON));
  assert_true(is_equal_ndarray(y2, y2_hat, NDARRAY_TYPE_EPSILON));

  free_ndarray(&x);
  free_ndarray(&y0);
  free_ndarray(&y1);
  free_ndarray(&y2);
  free_ndarray(&y0_hat);
  free_ndarray(&y1_hat);
  free_ndarray(&y2_hat);
}

int main(void) {
  const struct CMUnitTest tests[] = {
      cmocka_unit_test(test_full_ndarray),
      cmocka_unit_test(test_copy_ndarray),
      cmocka_unit_test(test_empty_like_ndarray),
      cmocka_unit_test(test_zeros_ndarray),
      cmocka_unit_test(test_ones_ndarray),
      cmocka_unit_test(test_eye_ndarray),
      cmocka_unit_test(test_random_ndrray),
      cmocka_unit_test(test_random_truncated_ndarray),
      cmocka_unit_test(test_read_ndarray),
      cmocka_unit_test(test_is_equal),
      cmocka_unit_test(test_unary_op_ndarray),
      cmocka_unit_test(test_log_ndarray),
      cmocka_unit_test(test_add_ndarray_scalar),
      cmocka_unit_test(test_subtract_ndarray_scalar),
      cmocka_unit_test(test_multiply_ndarray_scalar),
      cmocka_unit_test(test_divide_ndarray_scalar),
      cmocka_unit_test(test_divide_scalar_ndarray),
      cmocka_unit_test(test_power_ndarray_scalar),
      cmocka_unit_test(test_binary_op_ndarray),
      cmocka_unit_test(test_add_ndarray_ndarray),
      cmocka_unit_test(test_subtract_ndarray_ndarray),
      cmocka_unit_test(test_multiply_ndarray_ndarray),
      cmocka_unit_test(test_divide_ndarray_ndarray),
      cmocka_unit_test(test_power_ndarray_ndarray),
      cmocka_unit_test(test_matmul_ndarray),
      cmocka_unit_test(test_transpose_ndarray),
      cmocka_unit_test(test_max_ndarray),
      cmocka_unit_test(test_min_ndarray),
      cmocka_unit_test(test_sum_ndarray),

  };
  return cmocka_run_group_tests(tests, NULL, NULL);
}
