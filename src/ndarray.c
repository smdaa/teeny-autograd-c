#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "ndarray.h"

NDARRAY_TYPE add(NDARRAY_TYPE a, NDARRAY_TYPE b) { return a + b; }

NDARRAY_TYPE subtract(NDARRAY_TYPE a, NDARRAY_TYPE b) { return a - b; }

NDARRAY_TYPE multiply(NDARRAY_TYPE a, NDARRAY_TYPE b) { return a * b; }

NDARRAY_TYPE divide(NDARRAY_TYPE a, NDARRAY_TYPE b) { return a / b; }

NDARRAY_TYPE divide_(NDARRAY_TYPE a, NDARRAY_TYPE b) { return b / a; }

NDARRAY_TYPE power(NDARRAY_TYPE a, NDARRAY_TYPE b) { return pow(a, b); }

int get_size(int dim, const int *shape)
{
    int size = 1;
    for (int i = 0; i < dim; i++)
    {
        size *= shape[i];
    }
    return size;
}

ndarray *full_ndarray(int dim, int *shape, NDARRAY_TYPE value)
{
    ndarray *arr = (ndarray *)malloc(sizeof(ndarray));
    arr->dim = dim;
    arr->size = get_size(dim, shape);
    arr->shape = (int *)malloc(dim * sizeof(int));
    for (int i = 0; i < dim; i++)
    {
        arr->shape[i] = shape[i];
    }
    arr->data = (NDARRAY_TYPE *)malloc(arr->size * sizeof(NDARRAY_TYPE));
    for (int i = 0; i < arr->size; i++)
    {
        arr->data[i] = value;
    }

    return arr;
}

ndarray *copy_ndarray(ndarray *arr)
{
    ndarray *arr_copy = (ndarray *)malloc(sizeof(ndarray));
    arr_copy->dim = arr->dim;
    arr_copy->size = arr->size;
    arr_copy->shape = (int *)malloc(arr->dim * sizeof(int));
    for (int i = 0; i < arr->dim; i++)
    {
        arr_copy->shape[i] = arr->shape[i];
    }
    arr_copy->data = (NDARRAY_TYPE *)malloc(arr->size * sizeof(NDARRAY_TYPE));
    for (int i = 0; i < arr->size; i++)
    {
        arr_copy->data[i] = arr->data[i];
    }

    return arr_copy;
}

ndarray *empty_like_ndarray(ndarray *arr)
{
    ndarray *n_arr = (ndarray *)malloc(sizeof(ndarray));
    n_arr->dim = arr->dim;
    n_arr->size = arr->size;
    n_arr->shape = (int *)malloc(arr->dim * sizeof(int));
    for (int i = 0; i < arr->dim; i++)
    {
        n_arr->shape[i] = arr->shape[i];
    }
    n_arr->data = (NDARRAY_TYPE *)malloc(arr->size * sizeof(NDARRAY_TYPE));

    return n_arr;
}

ndarray *zeros_ndarray(int dim, int *shape)
{
    return full_ndarray(dim, shape, 0.0);
}

ndarray *ones_ndarray(int dim, int *shape)
{
    return full_ndarray(dim, shape, 1.0);
}

ndarray *eye_ndarray(int size)
{
    ndarray *arr = zeros_ndarray(2, (int[]){size, size});
    int offset = 0;
    for (int i = 0; i < size * size; i += size)
    {
        arr->data[i + offset] = 1.0;
        offset++;
    }

    return arr;
}

ndarray *random_ndrray(int dim, int *shape)
{
    ndarray *arr = (ndarray *)malloc(sizeof(ndarray));
    arr->dim = dim;
    arr->size = get_size(dim, shape);
    arr->shape = (int *)malloc(dim * sizeof(int));
    for (int i = 0; i < dim; i++)
    {
        arr->shape[i] = shape[i];
    }
    arr->data = (NDARRAY_TYPE *)malloc(arr->size * sizeof(NDARRAY_TYPE));
    for (int i = 0; i < arr->size; i++)
    {
        arr->data[i] = (NDARRAY_TYPE)rand() / (RAND_MAX);
    }

    return arr;
}

ndarray *unary_op_ndarray(ndarray *arr, NDARRAY_TYPE (*op)(NDARRAY_TYPE))
{
    ndarray *n_arr = (ndarray *)malloc(sizeof(ndarray));
    n_arr->dim = arr->dim;
    n_arr->size = arr->size;
    n_arr->shape = (int *)malloc(arr->dim * sizeof(int));
    for (int i = 0; i < arr->dim; i++)
    {
        n_arr->shape[i] = arr->shape[i];
    }
    n_arr->data = (NDARRAY_TYPE *)malloc(arr->size * sizeof(NDARRAY_TYPE));
    for (int i = 0; i < arr->size; i++)
    {
        n_arr->data[i] = op(arr->data[i]);
    }

    return n_arr;
}

ndarray *log_ndarray(ndarray *arr)
{
    return unary_op_ndarray(arr, log);
}

ndarray *unary_op_ndarray_scalar(ndarray *arr, NDARRAY_TYPE scalar, NDARRAY_TYPE (*op)(NDARRAY_TYPE, NDARRAY_TYPE))
{
    ndarray *n_arr = (ndarray *)malloc(sizeof(ndarray));
    n_arr->dim = arr->dim;
    n_arr->size = arr->size;
    n_arr->shape = (int *)malloc(arr->dim * sizeof(int));
    for (int i = 0; i < arr->dim; i++)
    {
        n_arr->shape[i] = arr->shape[i];
    }
    n_arr->data = (NDARRAY_TYPE *)malloc(arr->size * sizeof(NDARRAY_TYPE));
    for (int i = 0; i < arr->size; i++)
    {
        n_arr->data[i] = op(arr->data[i], scalar);
    }

    return n_arr;
}

ndarray *add_ndarray_scalar(ndarray *arr, NDARRAY_TYPE scalar)
{
    return unary_op_ndarray_scalar(arr, scalar, add);
}

ndarray *subtract_ndarray_scalar(ndarray *arr, NDARRAY_TYPE scalar)
{
    return unary_op_ndarray_scalar(arr, scalar, subtract);
}

ndarray *multiply_ndarray_scalar(ndarray *arr, NDARRAY_TYPE scalar)
{
    return unary_op_ndarray_scalar(arr, scalar, multiply);
}

ndarray *divide_ndarray_scalar(ndarray *arr, NDARRAY_TYPE scalar)
{
    return unary_op_ndarray_scalar(arr, scalar, divide);
}

ndarray *divide_scalar_ndarray(ndarray *arr, NDARRAY_TYPE scalar)
{
    return unary_op_ndarray_scalar(arr, scalar, divide_);
}

ndarray *power_ndarray_scalar(ndarray *arr, NDARRAY_TYPE scalar)
{
    return unary_op_ndarray_scalar(arr, scalar, power);
}

ndarray *binary_op_ndarray(ndarray *arr1, ndarray *arr2, NDARRAY_TYPE (*op)(NDARRAY_TYPE, NDARRAY_TYPE))
{
    if (arr1->dim != arr2->dim)
    {
        printf("Incompatible dimensions");
        return NULL;
    }
    for (int i = 0; i < arr1->dim; i++)
    {
        if (arr1->shape[i] != arr2->shape[i])
        {
            printf("Incompatible shapes");
            return NULL;
        }
    }

    ndarray *arr = (ndarray *)malloc(sizeof(ndarray));
    arr->dim = arr1->dim;
    arr->size = arr1->size;
    arr->shape = (int *)malloc(arr->dim * sizeof(int));
    for (int i = 0; i < arr->dim; i++)
    {
        arr->shape[i] = arr1->shape[i];
    }
    arr->data = (NDARRAY_TYPE *)malloc(arr->size * sizeof(NDARRAY_TYPE));
    for (int i = 0; i < arr->size; i++)
    {
        arr->data[i] = op(arr1->data[i], arr2->data[i]);
    }

    return arr;
}

ndarray *add_ndarray_ndarray(ndarray *arr1, ndarray *arr2)
{
    return binary_op_ndarray(arr1, arr2, add);
}

ndarray *subtract_ndarray_ndarray(ndarray *arr1, ndarray *arr2)
{
    return binary_op_ndarray(arr1, arr2, subtract);
}

ndarray *multiply_ndarray_ndarray(ndarray *arr1, ndarray *arr2)
{
    return binary_op_ndarray(arr1, arr2, multiply);
}

ndarray *divide_ndarray_ndarray(ndarray *arr1, ndarray *arr2)
{
    return binary_op_ndarray(arr1, arr2, divide);
}

ndarray *power_ndarray_ndarray(ndarray *arr1, ndarray *arr2)
{
    return binary_op_ndarray(arr1, arr2, power);
}

void matmul_2d_ndarray_helper(NDARRAY_TYPE *a, NDARRAY_TYPE *b, NDARRAY_TYPE *c, int p, int q, int r)
{
    for (int i = 0; i < p; i++)
    {
        for (int k = 0; k < q; k++)
        {
            register NDARRAY_TYPE v = a[i * q + k];
            for (int j = 0; j < r; j++)
            {
                c[i * r + j] += v * b[k * r + j];
            }
        }
    }
}

int get_offset(ndarray *arr, int *position, int pdim)
{
    unsigned int offset = 0;
    unsigned int len = arr->size;
    for (int i = 0; i < pdim; i++)
    {
        len /= arr->shape[i];
        offset += position[i] * len;
    }
    return offset;
}

void matmul_ndarray_helper(ndarray *arr, ndarray *arr1, ndarray *arr2, int *position, int dim)
{
    if (dim >= arr->dim - 2)
    {
        int position1[arr1->dim - 2];
        int position2[arr2->dim - 2];

        int offset1 = (arr1->dim >= arr2->dim) ? 0 : arr2->dim - arr1->dim;
        int offset2 = (arr1->dim >= arr2->dim) ? arr1->dim - arr2->dim : 0;

        for (int i = 0; i < arr1->dim - 2; i++)
        {
            position1[i] = position[offset1 + i];
        }
        for (int i = 0; i < arr2->dim - 2; i++)
        {
            position2[i] = position[offset2 + i];
        }

        NDARRAY_TYPE *mat1 = arr1->data + get_offset(arr1, position1, arr1->dim - 2);
        NDARRAY_TYPE *mat2 = arr2->data + get_offset(arr2, position2, arr2->dim - 2);
        NDARRAY_TYPE *mat = arr->data + get_offset(arr, position, dim);

        int p = arr1->shape[arr1->dim - 2];
        int q = arr1->shape[arr1->dim - 1];
        int r = arr2->shape[arr2->dim - 1];

        matmul_2d_ndarray_helper(mat1, mat2, mat, p, q, r);
        return;
    }

    for (int i = 0; i < arr->shape[dim]; i++)
    {
        position[dim] = i;
        matmul_ndarray_helper(arr, arr1, arr2, position, dim + 1);
    }
}

ndarray *matmul_ndarray(ndarray *arr1, ndarray *arr2)
{
    int bound = (arr1->dim >= arr2->dim) ? arr2->dim - 2 : arr1->dim - 2;
    for (int i = 0; i < bound; i++)
    {
        if (arr1->shape[arr1->dim - i - 3] != arr2->shape[arr2->dim - i - 3])
        {
            printf("Incompatible shapes\n");
            return NULL;
        }
    }

    int dim;
    int *shape;
    if (arr1->dim >= arr2->dim)
    {
        dim = arr1->dim;
        shape = (int *)malloc(dim * sizeof(int));
        for (int i = 0; i < dim - 1; i++)
        {
            shape[i] = arr1->shape[i];
        }
        shape[dim - 1] = arr2->shape[arr2->dim - 1];
    }
    else
    {
        dim = arr2->dim;
        shape = (int *)malloc(dim * sizeof(int));
        for (int i = 0; i < dim - 1; i++)
        {
            shape[i] = arr2->shape[i];
        }
        shape[dim - 1] = arr1->shape[arr1->dim - 1];
    }

    ndarray *arr = zeros_ndarray(dim, shape);
    free(shape);
    int position[dim - 2];
    for (int i = 0; i < dim - 2; i++)
    {
        position[i] = 0;
    }

    matmul_ndarray_helper(arr, arr1, arr2, position, 0);

    return arr;
}

void transpose_ndarray_helper(ndarray *arr, ndarray *n_arr, int *position, int *order, int dim)
{
    if (dim >= arr->dim)
    {
        int tdim = n_arr->dim;
        int n_position[tdim];

        for (int i = 0; i < tdim; i++)
        {
            n_position[i] = position[order[i]];
        }

        int offset_arr = get_offset(arr, position, arr->dim);
        int offset_narr = get_offset(n_arr, n_position, n_arr->dim);

        n_arr->data[offset_narr] = arr->data[offset_arr];

        return;
    }
    for (int i = 0; i < arr->shape[dim]; i++)
    {
        position[dim] = i;
        transpose_ndarray_helper(arr, n_arr, position, order, dim + 1);
    }
}

ndarray *transpose_ndarray(ndarray *arr, int *order)
{
    int *shape = (int *)malloc(arr->dim * sizeof(int));
    for (int i = 0; i < arr->dim; i++)
    {
        shape[i] = arr->shape[order[i]];
    }
    ndarray *n_arr = zeros_ndarray(arr->dim, shape);
    free(shape);
    int position[arr->dim];
    transpose_ndarray_helper(arr, n_arr, position, order, 0);

    return n_arr;
}

ndarray *read_ndarray(const char *filename)
{
    FILE *file = fopen(filename, "r");
    if (file == NULL)
    {
        perror("Error opening file");
        return NULL;
    }
    ndarray *arr = (ndarray *)malloc(sizeof(ndarray));
    fscanf(file, "%d %d", &arr->dim, &arr->size);
    arr->shape = (int *)malloc(arr->dim * sizeof(int));
    arr->data = (NDARRAY_TYPE *)malloc(arr->size * sizeof(NDARRAY_TYPE));
    for (int i = 0; i < arr->dim; i++)
    {
        fscanf(file, "%d", &arr->shape[i]);
    }
    for (int i = 0; i < arr->size; i++)
    {
        fscanf(file, "%le", &arr->data[i]);
    }
    fclose(file);

    return arr;
}

bool is_equal(ndarray *arr1, ndarray *arr2)
{
    if (arr1->dim != arr2->dim || arr1->size != arr2->size)
    {
        return false;
    }
    for (int i = 0; i < arr1->dim; i++)
    {
        if (arr1->shape[i] != arr2->shape[i])
        {
            return false;
        }
    }
    for (int i = 0; i < arr1->size; i++)
    {
        if (fabs(arr1->data[i] - arr2->data[i]) > EPSILON)
        {
            return false;
        }
    }

    return true;
}

void print_ndarray_helper(ndarray *arr, int level, int *index, int tab)
{
    for (int i = 0; i < tab; i++)
    {
        printf("\t");
    }
    printf("[");
    if (level == arr->dim - 1)
    {
        for (int i = 0; i < arr->shape[level]; i++)
        {
            printf("%.2lf", arr->data[*index]);
            if (i != arr->shape[level] - 1)
            {
                printf(", ");
            }
            (*index)++;
        }
        printf("]");
    }
    else
    {
        printf("\n");
        for (int i = 0; i < arr->shape[level]; i++)
        {
            print_ndarray_helper(arr, level + 1, index, tab + 1);
            if (i != arr->shape[level] - 1)
            {
                printf(",\n");
            }
        }
        printf("\n");
        for (int i = 0; i < tab; i++)
        {
            printf("\t");
        }
        printf("]");
    }
}

void print_ndarray(ndarray *arr)
{
    int index = 0;
    print_ndarray_helper(arr, 0, &index, 0);
    printf("\n");

    printf("Shape: (");
    for (int i = 0; i < arr->dim; i++)
    {
        printf("%d", arr->shape[i]);
        if (i != arr->dim - 1)
        {
            printf(", ");
        }
    }
    printf(")\n");
}

void free_ndarray(ndarray **arr)
{
    if (*arr == NULL)
    {
        return;
    }

    if ((*arr)->shape != NULL)
    {
        free((*arr)->shape);
        (*arr)->shape = NULL;
    }

    if ((*arr)->data != NULL)
    {
        free((*arr)->data);
        (*arr)->data = NULL;
    }

    free(*arr);
    *arr = NULL;
}
