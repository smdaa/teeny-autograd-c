#include <stdint.h>
#include <stddef.h>
#include <math.h>
#include <setjmp.h>
#include <stdarg.h>
#include "../extern/cmocka/include/cmocka.h"
#include "../src/ndarray.h"

// valgrind --leak-check=full --show-leak-kinds=all --errors-for-leak-kinds=all
// --error-exitcode=1 ./test_ndarray

const char *dataDir;

static void test_full_ndarray(void **state) {
    (void) state;

    ndarray *arr = full_ndarray(2, (int[]) {2, 3}, 5.0);

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
    (void) state;

    ndarray *src = full_ndarray(2, (int[]) {2, 3}, 5);
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
    (void) state;

    ndarray *src = full_ndarray(2, (int[]) {2, 3}, 5);
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
    (void) state;

    ndarray *arr = zeros_ndarray(2, (int[]) {2, 3});

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
    (void) state;

    ndarray *arr = ones_ndarray(2, (int[]) {2, 3});

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
    (void) state;

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

static void test_random_uniform_ndarray(void **state) {
    (void) state;
    NDARRAY_TYPE min_val = -0.1f;
    NDARRAY_TYPE max_val = 0.1f;
    ndarray *arr = random_uniform_ndarray(2, (int[]) {2, 1000}, min_val, max_val);
    assert_non_null(arr);
    assert_int_equal(arr->dim, 2);
    assert_int_equal(arr->size, 2000);
    assert_int_equal(arr->shape[0], 2);
    assert_int_equal(arr->shape[1], 1000);
    for (int i = 0; i < arr->size; i++) {
        assert_true(arr->data[i] > min_val && arr->data[i] < max_val);
    }

    free_ndarray(&arr);
}

static void test_random_normal_ndarray(void **state) {
    (void) state;
    NDARRAY_TYPE mean = 1.0;
    NDARRAY_TYPE std = 2.0;
    ndarray *arr = random_normal_ndarray(2, (int[]) {1, 100000}, mean, std);
    assert_non_null(arr);
    assert_int_equal(arr->dim, 2);
    assert_int_equal(arr->size, 100000);
    assert_int_equal(arr->shape[0], 1);
    assert_int_equal(arr->shape[1], 100000);
    NDARRAY_TYPE mean_hat = 0.0;
    NDARRAY_TYPE std_hat = 0.0;
    for (int i = 0; i < arr->size; i++) {
        mean_hat += arr->data[i];
    }
    mean_hat /= (NDARRAY_TYPE)arr->size;
    for (int i = 0; i < arr->size; i++) {
        std_hat += (arr->data[i] - mean_hat) * (arr->data[i] - mean_hat);
    }
    std_hat = sqrt(std_hat / (NDARRAY_TYPE)arr->size);
    assert_double_equal(mean_hat, mean , 1E-2);
    assert_double_equal(std_hat, std , 1E-2);

    free_ndarray(&arr);
}

static void test_random_truncated_normal_ndarray(void **state) {
    (void) state;

    ndarray *arr =
            random_truncated_normal_ndarray(2, (int[]) {2, 3}, 0.0, 1.0, -0.5, 0.5);

    assert_non_null(arr);
    assert_int_equal(arr->dim, 2);
    assert_int_equal(arr->size, 6);
    assert_int_equal(arr->shape[0], 2);
    assert_int_equal(arr->shape[1], 3);
    for (int i = 0; i < arr->size; i++) {
        assert_true(arr->data[i] > -1.0f && arr->data[i] < 1.0f);
    }

    free_ndarray(&arr);
}

static void test_read_ndarray(void **state) {
    (void) state;

    char path[256];
    snprintf(path, sizeof(path), "%s/test_read_ndarray/x.txt", dataDir);
    ndarray *x = read_ndarray(path);

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
    (void) state;

    ndarray arr1 = {2, 4, (int[]) {2, 2}, (NDARRAY_TYPE[]) {1, 2, 3, 4}};
    ndarray arr2 = {2, 4, (int[]) {2, 2}, (NDARRAY_TYPE[]) {1, 2, 3, 4}};
    ndarray arr3 = {2, 4, (int[]) {2, 2}, (NDARRAY_TYPE[]) {1, 2, 3, 4}};
    ndarray arr4 = {3, 4, (int[]) {2, 2, 1}, (NDARRAY_TYPE[]) {1, 2, 3, 4}};
    ndarray arr5 = {2, 4, (int[]) {2, 2}, (NDARRAY_TYPE[]) {1, 2, 3, 4}};
    ndarray arr6 = {2, 4, (int[]) {2, 2}, (NDARRAY_TYPE[]) {1, 2, 5, 4}};

    assert_true(is_equal_ndarray(&arr1, &arr2, NDARRAY_TYPE_EPSILON));
    assert_false(is_equal_ndarray(&arr3, &arr4, NDARRAY_TYPE_EPSILON));
    assert_false(is_equal_ndarray(&arr5, &arr6, NDARRAY_TYPE_EPSILON));
}

static void test_unary_op_ndarray(void **state) {
    (void) state;

    char path[256];
    snprintf(path, sizeof(path), "%s/test_unary_op_ndarray/x.txt", dataDir);
    ndarray *x = read_ndarray(path);
    snprintf(path, sizeof(path), "%s/test_unary_op_ndarray/y.txt", dataDir);
    ndarray *y = read_ndarray(path);
    ndarray *y_hat = unary_op_ndarray(x, sin);

    assert_true(is_equal_ndarray(y, y_hat, NDARRAY_TYPE_EPSILON));

    free_ndarray(&x);
    free_ndarray(&y);
    free_ndarray(&y_hat);
}

static void test_log_ndarray(void **state) {
    (void) state;

    char path[256];
    snprintf(path, sizeof(path), "%s/test_log_ndarray/x.txt", dataDir);
    ndarray *x = read_ndarray(path);
    snprintf(path, sizeof(path), "%s/test_log_ndarray/y.txt", dataDir);
    ndarray *y = read_ndarray(path);
    ndarray *y_hat = log_ndarray(x);

    assert_true(is_equal_ndarray(y, y_hat, NDARRAY_TYPE_EPSILON));

    free_ndarray(&x);
    free_ndarray(&y);
    free_ndarray(&y_hat);
}

static void test_add_ndarray_scalar(void **state) {
    (void) state;

    char path[256];
    snprintf(path, sizeof(path), "%s/test_add_ndarray_scalar/x.txt", dataDir);
    ndarray *x = read_ndarray(path);
    snprintf(path, sizeof(path), "%s/test_add_ndarray_scalar/y.txt", dataDir);
    ndarray *y = read_ndarray(path);
    ndarray *y_hat = add_ndarray_scalar(x, 10);

    assert_true(is_equal_ndarray(y, y_hat, NDARRAY_TYPE_EPSILON));

    free_ndarray(&x);
    free_ndarray(&y);
    free_ndarray(&y_hat);
}

static void test_subtract_ndarray_scalar(void **state) {
    (void) state;

    char path[256];
    snprintf(path, sizeof(path), "%s/test_subtract_ndarray_scalar/x.txt",
             dataDir);
    ndarray *x = read_ndarray(path);
    snprintf(path, sizeof(path), "%s/test_subtract_ndarray_scalar/y.txt",
             dataDir);
    ndarray *y = read_ndarray(path);
    ndarray *y_hat = subtract_ndarray_scalar(x, 10);

    assert_true(is_equal_ndarray(y, y_hat, NDARRAY_TYPE_EPSILON));

    free_ndarray(&x);
    free_ndarray(&y);
    free_ndarray(&y_hat);
}

static void test_multiply_ndarray_scalar(void **state) {
    (void) state;

    char path[256];
    snprintf(path, sizeof(path), "%s/test_multiply_ndarray_scalar/x.txt",
             dataDir);
    ndarray *x = read_ndarray(path);
    snprintf(path, sizeof(path), "%s/test_multiply_ndarray_scalar/y.txt",
             dataDir);
    ndarray *y = read_ndarray(path);
    ndarray *y_hat = multiply_ndarray_scalar(x, 10);

    assert_true(is_equal_ndarray(y, y_hat, NDARRAY_TYPE_EPSILON));

    free_ndarray(&x);
    free_ndarray(&y);
    free_ndarray(&y_hat);
}

static void test_divide_ndarray_scalar(void **state) {
    (void) state;

    char path[256];
    snprintf(path, sizeof(path), "%s/test_divide_ndarray_scalar/x.txt", dataDir);
    ndarray *x = read_ndarray(path);
    snprintf(path, sizeof(path), "%s/test_divide_ndarray_scalar/y.txt", dataDir);
    ndarray *y = read_ndarray(path);
    ndarray *y_hat = divide_ndarray_scalar(x, 10);

    assert_true(is_equal_ndarray(y, y_hat, NDARRAY_TYPE_EPSILON));

    free_ndarray(&x);
    free_ndarray(&y);
    free_ndarray(&y_hat);
}

static void test_divide_scalar_ndarray(void **state) {
    (void) state;

    char path[256];
    snprintf(path, sizeof(path), "%s/test_divide_scalar_ndarray/x.txt", dataDir);
    ndarray *x = read_ndarray(path);
    snprintf(path, sizeof(path), "%s/test_divide_scalar_ndarray/y.txt", dataDir);
    ndarray *y = read_ndarray(path);
    ndarray *y_hat = divide_scalar_ndarray(x, 10);

    assert_true(is_equal_ndarray(y, y_hat, NDARRAY_TYPE_EPSILON));

    free_ndarray(&x);
    free_ndarray(&y);
    free_ndarray(&y_hat);
}

static void test_power_ndarray_scalar(void **state) {
    (void) state;

    char path[256];
    snprintf(path, sizeof(path), "%s/test_power_ndarray_scalar/x.txt", dataDir);
    ndarray *x = read_ndarray(path);
    snprintf(path, sizeof(path), "%s/test_power_ndarray_scalar/y.txt", dataDir);
    ndarray *y = read_ndarray(path);
    ndarray *y_hat = power_ndarray_scalar(x, 2);

    assert_true(is_equal_ndarray(y, y_hat, NDARRAY_TYPE_EPSILON));

    free_ndarray(&x);
    free_ndarray(&y);
    free_ndarray(&y_hat);
}

static void test_binary_op_ndarray(void **state) {
    (void) state;

    char path[256];
    ndarray *a[6], *b[6], *c[6], *c_hat[6];

    for (int i = 0; i < 6; ++i) {
        snprintf(path, sizeof(path), "%s/test_binary_op_ndarray/a%d.txt", dataDir,
                 i);
        a[i] = read_ndarray(path);

        snprintf(path, sizeof(path), "%s/test_binary_op_ndarray/b%d.txt", dataDir,
                 i);
        b[i] = read_ndarray(path);

        snprintf(path, sizeof(path), "%s/test_binary_op_ndarray/c%d.txt", dataDir,
                 i);
        c[i] = read_ndarray(path);

        snprintf(path, sizeof(path), "%s/test_binary_op_ndarray/c%d_hat.txt",
                 dataDir, i);
        c_hat[i] = binary_op_ndarray(a[i], b[i], fmax);

        assert_true(is_equal_ndarray(c[i], c_hat[i], NDARRAY_TYPE_EPSILON));

        free_ndarray(&a[i]);
        free_ndarray(&b[i]);
        free_ndarray(&c[i]);
        free_ndarray(&c_hat[i]);
    }
}

static void test_add_ndarray_ndarray(void **state) {
    (void) state;

    char path[256];
    ndarray *a[6], *b[6], *c[6], *c_hat[6];

    for (int i = 0; i < 6; ++i) {
        snprintf(path, sizeof(path), "%s/test_add_ndarray_ndarray/a%d.txt", dataDir,
                 i);
        a[i] = read_ndarray(path);

        snprintf(path, sizeof(path), "%s/test_add_ndarray_ndarray/b%d.txt", dataDir,
                 i);
        b[i] = read_ndarray(path);

        snprintf(path, sizeof(path), "%s/test_add_ndarray_ndarray/c%d.txt", dataDir,
                 i);
        c[i] = read_ndarray(path);

        snprintf(path, sizeof(path), "%s/test_add_ndarray_ndarray/c%d_hat.txt",
                 dataDir, i);
        c_hat[i] = add_ndarray_ndarray(a[i], b[i]);

        assert_true(is_equal_ndarray(c[i], c_hat[i], NDARRAY_TYPE_EPSILON));

        free_ndarray(&a[i]);
        free_ndarray(&b[i]);
        free_ndarray(&c[i]);
        free_ndarray(&c_hat[i]);
    }
}

static void test_subtract_ndarray_ndarray(void **state) {
    (void) state;

    char path[256];
    ndarray *a[6], *b[6], *c[6], *c_hat[6];

    for (int i = 0; i < 6; ++i) {
        snprintf(path, sizeof(path), "%s/test_subtract_ndarray_ndarray/a%d.txt",
                 dataDir, i);
        a[i] = read_ndarray(path);

        snprintf(path, sizeof(path), "%s/test_subtract_ndarray_ndarray/b%d.txt",
                 dataDir, i);
        b[i] = read_ndarray(path);

        snprintf(path, sizeof(path), "%s/test_subtract_ndarray_ndarray/c%d.txt",
                 dataDir, i);
        c[i] = read_ndarray(path);

        snprintf(path, sizeof(path), "%s/test_subtract_ndarray_ndarray/c%d_hat.txt",
                 dataDir, i);
        c_hat[i] = subtract_ndarray_ndarray(a[i], b[i]);

        assert_true(is_equal_ndarray(c[i], c_hat[i], NDARRAY_TYPE_EPSILON));

        free_ndarray(&a[i]);
        free_ndarray(&b[i]);
        free_ndarray(&c[i]);
        free_ndarray(&c_hat[i]);
    }
}

static void test_multiply_ndarray_ndarray(void **state) {
    (void) state;

    char path[256];
    ndarray *a[6], *b[6], *c[6], *c_hat[6];

    for (int i = 0; i < 6; ++i) {
        snprintf(path, sizeof(path), "%s/test_multiply_ndarray_ndarray/a%d.txt",
                 dataDir, i);
        a[i] = read_ndarray(path);

        snprintf(path, sizeof(path), "%s/test_multiply_ndarray_ndarray/b%d.txt",
                 dataDir, i);
        b[i] = read_ndarray(path);

        snprintf(path, sizeof(path), "%s/test_multiply_ndarray_ndarray/c%d.txt",
                 dataDir, i);
        c[i] = read_ndarray(path);

        snprintf(path, sizeof(path), "%s/test_multiply_ndarray_ndarray/c%d_hat.txt",
                 dataDir, i);
        c_hat[i] = multiply_ndarray_ndarray(a[i], b[i]);

        assert_true(is_equal_ndarray(c[i], c_hat[i], NDARRAY_TYPE_EPSILON));

        free_ndarray(&a[i]);
        free_ndarray(&b[i]);
        free_ndarray(&c[i]);
        free_ndarray(&c_hat[i]);
    }
}

static void test_divide_ndarray_ndarray(void **state) {
    (void) state;

    char path[256];
    ndarray *a[6], *b[6], *c[6], *c_hat[6];

    for (int i = 0; i < 6; ++i) {
        snprintf(path, sizeof(path), "%s/test_divide_ndarray_ndarray/a%d.txt",
                 dataDir, i);
        a[i] = read_ndarray(path);

        snprintf(path, sizeof(path), "%s/test_divide_ndarray_ndarray/b%d.txt",
                 dataDir, i);
        b[i] = read_ndarray(path);

        snprintf(path, sizeof(path), "%s/test_divide_ndarray_ndarray/c%d.txt",
                 dataDir, i);
        c[i] = read_ndarray(path);

        snprintf(path, sizeof(path), "%s/test_divide_ndarray_ndarray/c%d_hat.txt",
                 dataDir, i);
        c_hat[i] = divide_ndarray_ndarray(a[i], b[i]);

        assert_true(is_equal_ndarray(c[i], c_hat[i], NDARRAY_TYPE_EPSILON));

        free_ndarray(&a[i]);
        free_ndarray(&b[i]);
        free_ndarray(&c[i]);
        free_ndarray(&c_hat[i]);
    }
}

static void test_power_ndarray_ndarray(void **state) {
    (void) state;

    char path[256];
    ndarray *a[6], *b[6], *c[6], *c_hat[6];

    for (int i = 0; i < 6; ++i) {
        snprintf(path, sizeof(path), "%s/test_power_ndarray_ndarray/a%d.txt",
                 dataDir, i);
        a[i] = read_ndarray(path);

        snprintf(path, sizeof(path), "%s/test_power_ndarray_ndarray/b%d.txt",
                 dataDir, i);
        b[i] = read_ndarray(path);

        snprintf(path, sizeof(path), "%s/test_power_ndarray_ndarray/c%d.txt",
                 dataDir, i);
        c[i] = read_ndarray(path);

        snprintf(path, sizeof(path), "%s/test_power_ndarray_ndarray/c%d_hat.txt",
                 dataDir, i);
        c_hat[i] = power_ndarray_ndarray(a[i], b[i]);

        assert_true(is_equal_ndarray(c[i], c_hat[i], NDARRAY_TYPE_EPSILON));

        free_ndarray(&a[i]);
        free_ndarray(&b[i]);
        free_ndarray(&c[i]);
        free_ndarray(&c_hat[i]);
    }
}

static void test_matmul_ndarray(void **state) {
    (void) state;

    char path[256];
    ndarray *a[4], *b[4], *c[4], *c_hat[4];

    for (int i = 0; i < 4; ++i) {
        snprintf(path, sizeof(path), "%s/test_matmul_ndarray/a%d.txt", dataDir, i);
        a[i] = read_ndarray(path);

        snprintf(path, sizeof(path), "%s/test_matmul_ndarray/b%d.txt", dataDir, i);
        b[i] = read_ndarray(path);

        snprintf(path, sizeof(path), "%s/test_matmul_ndarray/c%d.txt", dataDir, i);
        c[i] = read_ndarray(path);

        snprintf(path, sizeof(path), "%s/test_matmul_ndarray/c%d_hat.txt", dataDir,
                 i);
        c_hat[i] = matmul_ndarray(a[i], b[i]);

        assert_true(is_equal_ndarray(c[i], c_hat[i], NDARRAY_TYPE_EPSILON));

        free_ndarray(&a[i]);
        free_ndarray(&b[i]);
        free_ndarray(&c[i]);
        free_ndarray(&c_hat[i]);
    }
}

static void test_transpose_ndarray(void **state) {
    (void) state;

    char path[256];
    snprintf(path, sizeof(path), "%s/test_transpose_ndarray/x.txt", dataDir);
    ndarray *x = read_ndarray(path);

    int axes[3][3] = {{1, 0, 2},
                      {2, 1, 0},
                      {0, 2, 1}};
    ndarray *y_hat[3];

    for (int i = 0; i < 3; ++i) {
        snprintf(path, sizeof(path), "%s/test_transpose_ndarray/y%d.txt", dataDir,
                 i);
        ndarray *y = read_ndarray(path);
        y_hat[i] = transpose_ndarray(x, axes[i]);

        assert_true(is_equal_ndarray(y, y_hat[i], NDARRAY_TYPE_EPSILON));

        free_ndarray(&y);
    }

    free_ndarray(&x);

    for (int i = 0; i < 3; ++i) {
        free_ndarray(&y_hat[i]);
    }
}

static void test_max_ndarray(void **state) {
    (void) state;

    char path[256];
    snprintf(path, sizeof(path), "%s/test_max_ndarray/x.txt", dataDir);
    ndarray *x = read_ndarray(path);

    int axes[3] = {0, 1, 2};
    ndarray *y_hat[3];

    for (int i = 0; i < 3; ++i) {
        snprintf(path, sizeof(path), "%s/test_max_ndarray/y%d.txt", dataDir, i);
        ndarray *y = read_ndarray(path);
        y_hat[i] = max_ndarray(x, axes[i]);

        assert_true(is_equal_ndarray(y, y_hat[i], NDARRAY_TYPE_EPSILON));

        free_ndarray(&y);
    }

    free_ndarray(&x);

    for (int i = 0; i < 3; ++i) {
        free_ndarray(&y_hat[i]);
    }
}

static void test_min_ndarray(void **state) {
    (void) state;

    char path[256];
    snprintf(path, sizeof(path), "%s/test_min_ndarray/x.txt", dataDir);
    ndarray *x = read_ndarray(path);

    int axes[3] = {0, 1, 2};
    ndarray *y_hat[3];

    for (int i = 0; i < 3; ++i) {
        snprintf(path, sizeof(path), "%s/test_min_ndarray/y%d.txt", dataDir, i);
        ndarray *y = read_ndarray(path);
        y_hat[i] = min_ndarray(x, axes[i]);

        assert_true(is_equal_ndarray(y, y_hat[i], NDARRAY_TYPE_EPSILON));

        free_ndarray(&y);
    }

    free_ndarray(&x);

    for (int i = 0; i < 3; ++i) {
        free_ndarray(&y_hat[i]);
    }
}

static void test_sum_ndarray(void **state) {
    (void) state;

    char path[256];
    snprintf(path, sizeof(path), "%s/test_sum_ndarray/x.txt", dataDir);
    ndarray *x = read_ndarray(path);

    int axes[3] = {0, 1, 2};
    ndarray *y_hat[3];

    for (int i = 0; i < 3; ++i) {
        snprintf(path, sizeof(path), "%s/test_sum_ndarray/y%d.txt", dataDir, i);
        ndarray *y = read_ndarray(path);
        y_hat[i] = sum_ndarray(x, axes[i]);

        assert_true(is_equal_ndarray(y, y_hat[i], NDARRAY_TYPE_EPSILON));

        free_ndarray(&y);
    }

    free_ndarray(&x);

    for (int i = 0; i < 3; ++i) {
        free_ndarray(&y_hat[i]);
    }
}

int main(void) {
    dataDir = getenv("TEST_NDARRAY_DATA_DIR");
    if (dataDir == NULL) {
        fprintf(stderr,
                "Error: TEST_NDARRAY_DATA_DIR environment variable not set.\n");
        return 1;
    }
    const struct CMUnitTest tests[] = {
            cmocka_unit_test(test_full_ndarray),
            cmocka_unit_test(test_copy_ndarray),
            cmocka_unit_test(test_empty_like_ndarray),
            cmocka_unit_test(test_zeros_ndarray),
            cmocka_unit_test(test_ones_ndarray),
            cmocka_unit_test(test_eye_ndarray),
            cmocka_unit_test(test_random_uniform_ndarray),
            cmocka_unit_test(test_random_normal_ndarray),
            cmocka_unit_test(test_random_truncated_normal_ndarray),
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
