#include <stdbool.h>
#include <float.h>
#ifndef TEENY_AUTOGRAD_C_NDARRAY_H
#define TEENY_AUTOGRAD_C_NDARRAY_H

#define NDARRAY_TYPE double
#define NDARRAY_TYPE_MIN DBL_MIN
#define NDARRAY_TYPE_MAX DBL_MAX
#define NDARRAY_TYPE_EPSILON 1E-10

typedef struct ndarray
{
    int dim;
    int size;
    int *shape;
    NDARRAY_TYPE *data;
} ndarray;

ndarray *full_ndarray(int dim, int *shape, NDARRAY_TYPE value);

ndarray *copy_ndarray(ndarray *arr);

ndarray *empty_like_ndarray(ndarray *arr);

ndarray *zeros_ndarray(int dim, int *shape);

ndarray *ones_ndarray(int dim, int *shape);

ndarray *eye_ndarray(int size);

ndarray *random_ndrray(int dim, int *shape);

ndarray *read_ndarray(const char *filename);

bool is_equal_ndarray(ndarray *arr1, ndarray *arr2, double tolerance);

ndarray *unary_op_ndarray(ndarray *arr, NDARRAY_TYPE (*op)(NDARRAY_TYPE));

ndarray *log_ndarray(ndarray *arr);

ndarray *unary_op_ndarray_scalar(ndarray *arr, NDARRAY_TYPE scalar, NDARRAY_TYPE (*op)(NDARRAY_TYPE, NDARRAY_TYPE));

ndarray *add_ndarray_scalar(ndarray *arr, NDARRAY_TYPE scalar);

ndarray *subtract_ndarray_scalar(ndarray *arr, NDARRAY_TYPE scalar);

ndarray *multiply_ndarray_scalar(ndarray *arr, NDARRAY_TYPE scalar);

ndarray *divide_ndarray_scalar(ndarray *arr, NDARRAY_TYPE scalar);

ndarray *divide_scalar_ndarray(ndarray *arr, NDARRAY_TYPE scalar);

ndarray *power_ndarray_scalar(ndarray *arr, NDARRAY_TYPE scalar);

ndarray *binary_op_ndarray(ndarray *arr1, ndarray *arr2, NDARRAY_TYPE (*op)(NDARRAY_TYPE, NDARRAY_TYPE));

ndarray *add_ndarray_ndarray(ndarray *arr1, ndarray *arr2);

ndarray *subtract_ndarray_ndarray(ndarray *arr1, ndarray *arr2);

ndarray *multiply_ndarray_ndarray(ndarray *arr1, ndarray *arr2);

ndarray *divide_ndarray_ndarray(ndarray *arr1, ndarray *arr2);

ndarray *power_ndarray_ndarray(ndarray *arr1, ndarray *arr2);

ndarray *matmul_ndarray(ndarray *arr1, ndarray *arr2);

ndarray *transpose_ndarray(ndarray *arr, int *order);

ndarray *reduce_ndarray(ndarray *arr, NDARRAY_TYPE (*op)(NDARRAY_TYPE, NDARRAY_TYPE), int axis, NDARRAY_TYPE initial_value);

ndarray *max_ndarray(ndarray *arr, int axis);

ndarray *min_ndarray(ndarray *arr, int axis);

ndarray *sum_ndarray(ndarray *arr, int axis);

NDARRAY_TYPE reduce_all_ndarray(ndarray *arr, NDARRAY_TYPE (*op)(NDARRAY_TYPE, NDARRAY_TYPE), NDARRAY_TYPE initial_value);

NDARRAY_TYPE max_all_ndarray(ndarray *arr);

NDARRAY_TYPE min_all_ndarray(ndarray *arr);

NDARRAY_TYPE sum_all_ndarray(ndarray *arr);

void print_ndarray(ndarray *arr);

void free_ndarray(ndarray **arr);

#endif