#include <stdbool.h>
#ifndef TEENY_AUTOGRAD_C_NDARRAY_H
#define TEENY_AUTOGRAD_C_NDARRAY_H

#define NDARRAY_TYPE double
#define EPSILON 0.0001

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

ndarray *transpose_ndarray(ndarray *arr, int* order);

ndarray *read_ndarray(const char *filename);

bool is_equal(ndarray* arr1, ndarray* arr2);

void print_ndarray(ndarray *arr);

void free_ndarray(ndarray **arr);

#endif