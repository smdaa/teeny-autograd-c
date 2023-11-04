#include <stdlib.h>
#include <stdarg.h>
#include <stddef.h>
#include <setjmp.h>
#include <cmocka.h>
#include <math.h>
#include "../src/ndarray.h"

#define EPSILON 0.0001

static void test_full_ndarray(void **state)
{
    (void)state;

    ndarray *arr = full_ndarray(2, (int[]){2, 3}, 5);

    assert_non_null(arr);
    assert_int_equal(arr->dim, 2);
    assert_int_equal(arr->size, 6);
    assert_int_equal(arr->shape[0], 2);
    assert_int_equal(arr->shape[1], 3);
    for (int i = 0; i < arr->size; i++)
    {
        assert_float_equal(arr->data[i], 5, EPSILON);
    }

    free(arr->shape);
    free(arr->data);
    free(arr);
}

static void test_copy_ndarray(void **state)
{
    (void)state;

    ndarray *src = full_ndarray(2, (int[]){2, 3}, 5);
    ndarray *dst = copy_ndarray(src);

    assert_non_null(dst);
    assert_int_equal(dst->dim, src->dim);
    assert_int_equal(dst->size, src->size);
    assert_int_equal(dst->shape[0], src->shape[0]);
    assert_int_equal(dst->shape[1], src->shape[1]);
    for (int i = 0; i < dst->size; i++)
    {
        assert_float_equal(dst->data[i], src->data[i], EPSILON);
    }

    free(src->shape);
    free(src->data);
    free(src);
    free(dst->shape);
    free(dst->data);
    free(dst);
}

static void test_empty_like_ndarray(void **state)
{
    (void)state;

    ndarray *src = full_ndarray(2, (int[]){2, 3}, 5);
    ndarray *dst = empty_like_ndarray(src);

    assert_non_null(dst);
    assert_int_equal(dst->dim, src->dim);
    assert_int_equal(dst->size, src->size);
    assert_int_equal(dst->shape[0], src->shape[0]);
    assert_int_equal(dst->shape[1], src->shape[1]);

    free(src->shape);
    free(src->data);
    free(src);
    free(dst->shape);
    free(dst->data);
    free(dst);
}

static void test_zeros_ndarray(void **state)
{
    (void)state;

    ndarray *arr = zeros_ndarray(2, (int[]){2, 3});

    assert_non_null(arr);
    assert_int_equal(arr->dim, 2);
    assert_int_equal(arr->size, 6);
    assert_int_equal(arr->shape[0], 2);
    assert_int_equal(arr->shape[1], 3);
    for (int i = 0; i < arr->size; i++)
    {
        assert_float_equal(arr->data[i], 0.0, EPSILON);
    }

    free(arr->shape);
    free(arr->data);
    free(arr);
}

static void test_ones_ndarray(void **state)
{
    (void)state;

    ndarray *arr = ones_ndarray(2, (int[]){2, 3});

    assert_non_null(arr);
    assert_int_equal(arr->dim, 2);
    assert_int_equal(arr->size, 6);
    assert_int_equal(arr->shape[0], 2);
    assert_int_equal(arr->shape[1], 3);
    for (int i = 0; i < arr->size; i++)
    {
        assert_float_equal(arr->data[i], 1.0, EPSILON);
    }

    free(arr->shape);
    free(arr->data);
    free(arr);
}

static void test_eye_ndarray(void **state)
{
    (void)state;

    ndarray *arr = eye_ndarray(5);

    assert_non_null(arr);
    assert_int_equal(arr->dim, 2);
    assert_int_equal(arr->size, 25);
    assert_int_equal(arr->shape[0], 5);
    assert_int_equal(arr->shape[1], 5);
    for (int i = 0; i < 5; i++)
    {
        for (int j = 0; j < 5; j++)
        {
            if (i == j)
            {
                assert_float_equal(arr->data[i * 5 + j], 1.0, EPSILON);
            }
            else
            {
                assert_float_equal(arr->data[i * 5 + j], 0.0, EPSILON);
            }
        }
    }

    free(arr->shape);
    free(arr->data);
    free(arr);
}

static void test_random_ndrray(void **state)
{
    (void)state;

    ndarray *arr = random_ndrray(2, (int[]){2, 3});

    assert_non_null(arr);
    assert_int_equal(arr->dim, 2);
    assert_int_equal(arr->size, 6);
    assert_int_equal(arr->shape[0], 2);
    assert_int_equal(arr->shape[1], 3);
    for (int i = 0; i < arr->size; i++)
    {
        assert_in_range(arr->data[0], 0.0f, 1.0f);
    }

    free(arr->shape);
    free(arr->data);
    free(arr);
}

static void test_unary_op_ndarray(void **state)
{
    (void)state;

    ndarray *src = random_ndrray(2, (int[]){2, 3});
    ndarray *dst = unary_op_ndarray(src, exp);

    assert_non_null(dst);
    assert_int_equal(dst->dim, src->dim);
    assert_int_equal(dst->size, src->size);
    assert_int_equal(dst->shape[0], src->shape[0]);
    assert_int_equal(dst->shape[1], src->shape[1]);
    for (int i = 0; i < dst->size; i++)
    {
        assert_float_equal(dst->data[i], exp(src->data[i]), EPSILON);
    }

    free(src->shape);
    free(src->data);
    free(src);
    free(dst->shape);
    free(dst->data);
    free(dst);
}

static void test_log_ndarray(void **state)
{
    (void)state;

    ndarray *src = random_ndrray(2, (int[]){2, 3});
    ndarray *dst = log_ndarray(src);

    assert_non_null(dst);
    assert_int_equal(dst->dim, src->dim);
    assert_int_equal(dst->size, src->size);
    assert_int_equal(dst->shape[0], src->shape[0]);
    assert_int_equal(dst->shape[1], src->shape[1]);
    for (int i = 0; i < dst->size; i++)
    {
        assert_float_equal(dst->data[i], log(src->data[i]), EPSILON);
    }

    free(src->shape);
    free(src->data);
    free(src);
    free(dst->shape);
    free(dst->data);
    free(dst);
}

static void test_unary_op_ndarray_scalar(void **state)
{
    (void)state;

    ndarray *src = random_ndrray(2, (int[]){2, 3});
    ndarray *dst = unary_op_ndarray_scalar(src, 0.5f, fmax);

    assert_non_null(dst);
    assert_int_equal(dst->dim, src->dim);
    assert_int_equal(dst->size, src->size);
    assert_int_equal(dst->shape[0], src->shape[0]);
    assert_int_equal(dst->shape[1], src->shape[1]);
    for (int i = 0; i < dst->size; i++)
    {
        assert_float_equal(dst->data[i], fmax(src->data[i], 0.5f), EPSILON);
    }

    free(src->shape);
    free(src->data);
    free(src);
    free(dst->shape);
    free(dst->data);
    free(dst);
}

static void test_add_ndarray_scalar(void **state)
{
    (void)state;

    ndarray *src = random_ndrray(2, (int[]){2, 3});
    ndarray *dst = add_ndarray_scalar(src, 2.0f);

    assert_non_null(dst);
    assert_int_equal(dst->dim, src->dim);
    assert_int_equal(dst->size, src->size);
    assert_int_equal(dst->shape[0], src->shape[0]);
    assert_int_equal(dst->shape[1], src->shape[1]);
    for (int i = 0; i < dst->size; i++)
    {
        assert_float_equal(dst->data[i], (src->data[i] + 2.0f), EPSILON);
    }

    free(src->shape);
    free(src->data);
    free(src);
    free(dst->shape);
    free(dst->data);
    free(dst);
}

static void test_subtract_ndarray_scalar(void **state)
{
    (void)state;

    ndarray *src = random_ndrray(2, (int[]){2, 3});
    ndarray *dst = subtract_ndarray_scalar(src, 2.0f);

    assert_non_null(dst);
    assert_int_equal(dst->dim, src->dim);
    assert_int_equal(dst->size, src->size);
    assert_int_equal(dst->shape[0], src->shape[0]);
    assert_int_equal(dst->shape[1], src->shape[1]);
    for (int i = 0; i < dst->size; i++)
    {
        assert_float_equal(dst->data[i], (src->data[i] - 2.0f), EPSILON);
    }

    free(src->shape);
    free(src->data);
    free(src);
    free(dst->shape);
    free(dst->data);
    free(dst);
}

static void test_multiply_ndarray_scalar(void **state)
{
    (void)state;

    ndarray *src = random_ndrray(2, (int[]){2, 3});
    ndarray *dst = multiply_ndarray_scalar(src, 2.0f);

    assert_non_null(dst);
    assert_int_equal(dst->dim, src->dim);
    assert_int_equal(dst->size, src->size);
    assert_int_equal(dst->shape[0], src->shape[0]);
    assert_int_equal(dst->shape[1], src->shape[1]);
    for (int i = 0; i < dst->size; i++)
    {
        assert_float_equal(dst->data[i], (src->data[i] * 2.0f), EPSILON);
    }

    free(src->shape);
    free(src->data);
    free(src);
    free(dst->shape);
    free(dst->data);
    free(dst);
}

static void test_divide_ndarray_scalar(void **state)
{
    (void)state;

    ndarray *src = random_ndrray(2, (int[]){2, 3});
    ndarray *dst = divide_ndarray_scalar(src, 2.0f);

    assert_non_null(dst);
    assert_int_equal(dst->dim, src->dim);
    assert_int_equal(dst->size, src->size);
    assert_int_equal(dst->shape[0], src->shape[0]);
    assert_int_equal(dst->shape[1], src->shape[1]);
    for (int i = 0; i < dst->size; i++)
    {
        assert_float_equal(dst->data[i], (src->data[i] / 2.0f), EPSILON);
    }

    free(src->shape);
    free(src->data);
    free(src);
    free(dst->shape);
    free(dst->data);
    free(dst);
}

static void test_divide_scalar_ndarray(void **state)
{
    (void)state;

    ndarray *src = random_ndrray(2, (int[]){2, 3});
    ndarray *dst = divide_scalar_ndarray(src, 2.0f);

    assert_non_null(dst);
    assert_int_equal(dst->dim, src->dim);
    assert_int_equal(dst->size, src->size);
    assert_int_equal(dst->shape[0], src->shape[0]);
    assert_int_equal(dst->shape[1], src->shape[1]);
    for (int i = 0; i < dst->size; i++)
    {
        assert_float_equal(dst->data[i], (2.0f / src->data[i]), EPSILON);
    }

    free(src->shape);
    free(src->data);
    free(src);
    free(dst->shape);
    free(dst->data);
    free(dst);
}

static void test_power_ndarray_scalar(void **state)
{
    (void)state;

    ndarray *src = random_ndrray(2, (int[]){2, 3});
    ndarray *dst = power_ndarray_scalar(src, 2.0f);

    assert_non_null(dst);
    assert_int_equal(dst->dim, src->dim);
    assert_int_equal(dst->size, src->size);
    assert_int_equal(dst->shape[0], src->shape[0]);
    assert_int_equal(dst->shape[1], src->shape[1]);
    for (int i = 0; i < dst->size; i++)
    {
        assert_float_equal(dst->data[i], powf(src->data[i], 2.0f), EPSILON);
    }

    free(src->shape);
    free(src->data);
    free(src);
    free(dst->shape);
    free(dst->data);
    free(dst);
}

static void test_binary_op_ndarray(void **state)
{
    (void)state;

    ndarray *arr1 = random_ndrray(2, (int[]){2, 3});
    ndarray *arr2 = random_ndrray(2, (int[]){2, 3});
    ndarray *arr = binary_op_ndarray(arr1, arr2, fmax);

    assert_non_null(arr);
    assert_int_equal(arr->dim, arr1->dim);
    assert_int_equal(arr->size, arr1->size);
    assert_int_equal(arr->shape[0], arr1->shape[0]);
    assert_int_equal(arr->shape[1], arr1->shape[1]);
    for (int i = 0; i < arr->size; i++)
    {
        assert_float_equal(arr->data[i], fmax(arr1->data[i], arr2->data[i]), EPSILON);
    }

    free(arr1->shape);
    free(arr1->data);
    free(arr1);
    free(arr2->shape);
    free(arr2->data);
    free(arr2);
    free(arr->shape);
    free(arr->data);
    free(arr);
}

static void test_add_ndarray_ndarray(void **state)
{
    (void)state;

    ndarray *arr1 = random_ndrray(2, (int[]){2, 3});
    ndarray *arr2 = random_ndrray(2, (int[]){2, 3});
    ndarray *arr = add_ndarray_ndarray(arr1, arr2);

    assert_non_null(arr);
    assert_int_equal(arr->dim, arr1->dim);
    assert_int_equal(arr->size, arr1->size);
    assert_int_equal(arr->shape[0], arr1->shape[0]);
    assert_int_equal(arr->shape[1], arr1->shape[1]);
    for (int i = 0; i < arr->size; i++)
    {
        assert_float_equal(arr->data[i], (arr1->data[i] + arr2->data[i]), EPSILON);
    }

    free(arr1->shape);
    free(arr1->data);
    free(arr1);
    free(arr2->shape);
    free(arr2->data);
    free(arr2);
    free(arr->shape);
    free(arr->data);
    free(arr);
}

static void test_subtract_ndarray_ndarray(void **state)
{
    (void)state;

    ndarray *arr1 = random_ndrray(2, (int[]){2, 3});
    ndarray *arr2 = random_ndrray(2, (int[]){2, 3});
    ndarray *arr = subtract_ndarray_ndarray(arr1, arr2);

    assert_non_null(arr);
    assert_int_equal(arr->dim, arr1->dim);
    assert_int_equal(arr->size, arr1->size);
    assert_int_equal(arr->shape[0], arr1->shape[0]);
    assert_int_equal(arr->shape[1], arr1->shape[1]);
    for (int i = 0; i < arr->size; i++)
    {
        assert_float_equal(arr->data[i], (arr1->data[i] - arr2->data[i]), EPSILON);
    }

    free(arr1->shape);
    free(arr1->data);
    free(arr1);
    free(arr2->shape);
    free(arr2->data);
    free(arr2);
    free(arr->shape);
    free(arr->data);
    free(arr);
}

static void test_multiply_ndarray_ndarray(void **state)
{
    (void)state;

    ndarray *arr1 = random_ndrray(2, (int[]){2, 3});
    ndarray *arr2 = random_ndrray(2, (int[]){2, 3});
    ndarray *arr = multiply_ndarray_ndarray(arr1, arr2);

    assert_non_null(arr);
    assert_int_equal(arr->dim, arr1->dim);
    assert_int_equal(arr->size, arr1->size);
    assert_int_equal(arr->shape[0], arr1->shape[0]);
    assert_int_equal(arr->shape[1], arr1->shape[1]);
    for (int i = 0; i < arr->size; i++)
    {
        assert_float_equal(arr->data[i], (arr1->data[i] * arr2->data[i]), EPSILON);
    }

    free(arr1->shape);
    free(arr1->data);
    free(arr1);
    free(arr2->shape);
    free(arr2->data);
    free(arr2);
    free(arr->shape);
    free(arr->data);
    free(arr);
}

static void test_divide_ndarray_ndarray(void **state)
{
    (void)state;

    ndarray *arr1 = random_ndrray(2, (int[]){2, 3});
    ndarray *arr2 = random_ndrray(2, (int[]){2, 3});
    ndarray *arr = divide_ndarray_ndarray(arr1, arr2);

    assert_non_null(arr);
    assert_int_equal(arr->dim, arr1->dim);
    assert_int_equal(arr->size, arr1->size);
    assert_int_equal(arr->shape[0], arr1->shape[0]);
    assert_int_equal(arr->shape[1], arr1->shape[1]);
    for (int i = 0; i < arr->size; i++)
    {
        assert_float_equal(arr->data[i], (arr1->data[i] / arr2->data[i]), EPSILON);
    }

    free(arr1->shape);
    free(arr1->data);
    free(arr1);
    free(arr2->shape);
    free(arr2->data);
    free(arr2);
    free(arr->shape);
    free(arr->data);
    free(arr);
}

static void test_power_ndarray_ndarray(void **state)
{
    (void)state;

    ndarray *arr1 = random_ndrray(2, (int[]){2, 3});
    ndarray *arr2 = random_ndrray(2, (int[]){2, 3});
    ndarray *arr = power_ndarray_ndarray(arr1, arr2);

    assert_non_null(arr);
    assert_int_equal(arr->dim, arr1->dim);
    assert_int_equal(arr->size, arr1->size);
    assert_int_equal(arr->shape[0], arr1->shape[0]);
    assert_int_equal(arr->shape[1], arr1->shape[1]);
    for (int i = 0; i < arr->size; i++)
    {
        assert_float_equal(arr->data[i], powf(arr1->data[i], arr2->data[i]), EPSILON);
    }

    free(arr1->shape);
    free(arr1->data);
    free(arr1);
    free(arr2->shape);
    free(arr2->data);
    free(arr2);
    free(arr->shape);
    free(arr->data);
    free(arr);
}

static void test_matmul_ndarray(void **state)
{
    (void)state;

    ndarray *arr1 = ones_ndarray(3, (int[]){2, 5, 3});
    ndarray *arr2 = ones_ndarray(3, (int[]){2, 3, 5});
    ndarray *arr = matmul_ndarray(arr1, arr2);

    assert_non_null(arr);
    assert_int_equal(arr->dim, 3);
    assert_int_equal(arr->size, 50);
    assert_int_equal(arr->shape[0], 2);
    assert_int_equal(arr->shape[1], 5);
    assert_int_equal(arr->shape[2], 5);
    for (int i = 0; i < arr->size; i++)
    {
        assert_float_equal(arr->data[i], 3.0, EPSILON);
    }

    free(arr1->shape);
    free(arr1->data);
    free(arr1);
    free(arr2->shape);
    free(arr2->data);
    free(arr2);
    free(arr->shape);
    free(arr->data);
    free(arr);
}

static void test_transpose_ndarray(void **state)
{
    (void)state;

    ndarray *arr = ones_ndarray(3, (int[]){2, 5, 3});
    ndarray *arr_t = transpose_ndarray(arr, (int[]){0, 2, 1});

    assert_non_null(arr_t);
    assert_int_equal(arr_t->dim, 3);
    assert_int_equal(arr_t->size, 30);
    assert_int_equal(arr_t->shape[0], 2);
    assert_int_equal(arr_t->shape[1], 3);
    assert_int_equal(arr_t->shape[2], 5);
    for (int i = 0; i < arr->size; i++)
    {
        assert_float_equal(arr->data[i], 1.0, EPSILON);
    }

    free(arr->shape);
    free(arr->data);
    free(arr);
    free(arr_t->shape);
    free(arr_t->data);
    free(arr_t);
}

static void test_read_ndarray(void **state)
{
    (void)state;

    ndarray *arr = read_ndarray("./test_data/test_read_ndarray.txt");

    assert_non_null(arr);
    assert_int_equal(arr->dim, 3);
    assert_int_equal(arr->size, 12);
    assert_int_equal(arr->shape[0], 2);
    assert_int_equal(arr->shape[1], 2);
    assert_int_equal(arr->shape[2], 3);
    assert_float_equal(arr->data[0], 1.0, EPSILON);
    assert_float_equal(arr->data[1], 2.0, EPSILON);
    assert_float_equal(arr->data[2], 3.0, EPSILON);
    assert_float_equal(arr->data[3], 4.0, EPSILON);
    assert_float_equal(arr->data[4], 5.0, EPSILON);
    assert_float_equal(arr->data[5], 6.0, EPSILON);
    assert_float_equal(arr->data[6], 1.5, EPSILON);
    assert_float_equal(arr->data[7], 2.5, EPSILON);
    assert_float_equal(arr->data[8], 3.5, EPSILON);
    assert_float_equal(arr->data[9], 4.5, EPSILON);
    assert_float_equal(arr->data[10], 5.5, EPSILON);
    assert_float_equal(arr->data[11], 6.5, EPSILON);

    free(arr->shape);
    free(arr->data);
    free(arr);
}

static void test_is_equal(void **state)
{
    (void)state;

    ndarray arr1 = {2, 4, (int[]){2, 2}, (NDARRAY_TYPE[]){1, 2, 3, 4}};
    ndarray arr2 = {2, 4, (int[]){2, 2}, (NDARRAY_TYPE[]){1, 2, 3, 4}};
    assert_true(is_equal(&arr1, &arr2));

    ndarray arr3 = {2, 4, (int[]){2, 2}, (NDARRAY_TYPE[]){1, 2, 3, 4}};
    ndarray arr4 = {3, 4, (int[]){2, 2, 1}, (NDARRAY_TYPE[]){1, 2, 3, 4}};
    assert_false(is_equal(&arr3, &arr4));

    ndarray arr5 = {2, 4, (int[]){2, 2}, (NDARRAY_TYPE[]){1, 2, 3, 4}};
    ndarray arr6 = {2, 4, (int[]){2, 2}, (NDARRAY_TYPE[]){1, 2, 5, 4}};
    assert_false(is_equal(&arr5, &arr6));
}

int main(void)
{
    const struct CMUnitTest tests[] = {
        cmocka_unit_test(test_full_ndarray),
        cmocka_unit_test(test_copy_ndarray),
        cmocka_unit_test(test_empty_like_ndarray),
        cmocka_unit_test(test_zeros_ndarray),
        cmocka_unit_test(test_ones_ndarray),
        cmocka_unit_test(test_eye_ndarray),
        cmocka_unit_test(test_random_ndrray),
        cmocka_unit_test(test_unary_op_ndarray),
        cmocka_unit_test(test_log_ndarray),
        cmocka_unit_test(test_unary_op_ndarray_scalar),
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
        cmocka_unit_test(test_read_ndarray),
        cmocka_unit_test(test_is_equal),
    };
    return cmocka_run_group_tests(tests, NULL, NULL);
}