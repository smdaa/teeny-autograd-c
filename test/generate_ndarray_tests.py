import argparse
import os
import torch

from save_ndarray import save_ndarray_to_file


def generate_unary_op_test_data(test, output_dir, unary_op, x_shape):
    dir_path = os.path.join(output_dir, test)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    x = torch.rand(x_shape, dtype=torch.double)
    y = unary_op(x)
    save_ndarray_to_file(os.path.join(dir_path, "x.txt"), x.detach().numpy())
    save_ndarray_to_file(os.path.join(dir_path, "y.txt"), y.detach().numpy())


def generate_binary_op_test_data(test, output_dir, binary_op, a_shapes, b_shapes):
    dir_path = os.path.join(output_dir, test)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    for i, (a_shape, b_shape) in enumerate(zip(a_shapes, b_shapes)):
        a = torch.rand(a_shape, dtype=torch.double)
        b = torch.rand(b_shape, dtype=torch.double)
        c = binary_op(a, b)
        for label, tensor in zip(("a", "b", "c"), (a, b, c)):
            file_path = os.path.join(dir_path, f"{label}{i}.txt")
            save_ndarray_to_file(file_path, tensor.detach().numpy())


def generate_transpose_test_data(test, output_dir, x_shape):
    dir_path = os.path.join(output_dir, test)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    x = torch.rand(x_shape, dtype=torch.double)
    y0 = x.transpose(0, 1)
    y1 = x.transpose(0, 2)
    y2 = x.transpose(1, 2)
    save_ndarray_to_file(os.path.join(dir_path, "x.txt"), x.detach().numpy())
    save_ndarray_to_file(os.path.join(dir_path, "y0.txt"), y0.detach().numpy())
    save_ndarray_to_file(os.path.join(dir_path, "y1.txt"), y1.detach().numpy())
    save_ndarray_to_file(os.path.join(dir_path, "y2.txt"), y2.detach().numpy())


def generate_reduce_test_data(test, output_dir, x_shape, operation_type):
    assert len(x_shape) >= 3
    dir_path = os.path.join(output_dir, test)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    x = torch.rand(x_shape, dtype=torch.double)
    if operation_type == "max":
        y0, _ = x.max(0, keepdim=True)
        y1, _ = x.max(1, keepdim=True)
        y2, _ = x.max(2, keepdim=True)
    elif operation_type == "min":
        y0, _ = x.min(0, keepdim=True)
        y1, _ = x.min(1, keepdim=True)
        y2, _ = x.min(2, keepdim=True)
    elif operation_type == "sum":
        y0 = x.sum(0, keepdim=True)
        y1 = x.sum(1, keepdim=True)
        y2 = x.sum(2, keepdim=True)
    else:
        raise ValueError("Invalid operation type")
    save_ndarray_to_file(os.path.join(dir_path, "y0.txt"), y0.detach().numpy())
    save_ndarray_to_file(os.path.join(dir_path, "y1.txt"), y1.detach().numpy())
    save_ndarray_to_file(os.path.join(dir_path, "y2.txt"), y2.detach().numpy())
    save_ndarray_to_file(os.path.join(dir_path, "x.txt"), x.detach().numpy())


def main():
    parser = argparse.ArgumentParser(
        description="Generate test data for ndarray operations"
    )
    parser.add_argument(
        "--output_dir",
        default="./test_data/test_ndarray/",
        help="Directory path to save the .txt files",
    )
    args = parser.parse_args()
    torch.manual_seed(0)

    # test_read_ndarray
    dir_path = os.path.join(args.output_dir, "test_read_ndarray")
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    n = 100
    x = torch.eye(n, dtype=torch.double) + 1
    save_ndarray_to_file(os.path.join(dir_path, "x.txt"), x.detach().numpy())

    # test_unary_op_ndarray
    n = 100
    x_shape = (n, n)
    generate_unary_op_test_data(
        "test_unary_op_ndarray",
        args.output_dir,
        torch.sin,
        x_shape,
    )

    # test_log_ndarray
    n = 100
    x_shape = (n, n)
    generate_unary_op_test_data("test_log_ndarray", args.output_dir, torch.log, x_shape)

    # test_add_ndarray_scalar
    n = 100
    x_shape = (n, n)
    generate_unary_op_test_data(
        "test_add_ndarray_scalar", args.output_dir, lambda x: x + 10, x_shape
    )

    # test_subtract_ndarray_scalar
    n = 100
    x_shape = (n, n)
    generate_unary_op_test_data(
        "test_subtract_ndarray_scalar", args.output_dir, lambda x: x - 10, x_shape
    )

    # test_multiply_ndarray_scalar
    n = 100
    x_shape = (n, n)
    generate_unary_op_test_data(
        "test_multiply_ndarray_scalar", args.output_dir, lambda x: x * 10, x_shape
    )

    # test_divide_ndarray_scalar
    n = 100
    x_shape = (n, n)
    generate_unary_op_test_data(
        "test_divide_ndarray_scalar", args.output_dir, lambda x: x / 10, x_shape
    )

    # test_divide_scalar_ndarray
    n = 100
    x_shape = (n, n)
    generate_unary_op_test_data(
        "test_divide_scalar_ndarray", args.output_dir, lambda x: 10 / x, x_shape
    )

    # test_power_ndarray_scalar
    n = 100
    x_shape = (n, n)
    generate_unary_op_test_data(
        "test_power_ndarray_scalar", args.output_dir, lambda x: x**2, x_shape
    )

    # test_binary_op_ndarray
    n = 10
    a_shapes = [(n, n), (n, n), (n, n), (n, n, 1), (1, n, n, n), (n, 1, n, n, n)]
    b_shapes = [(n, n), (1, n), (n, 1), (1, n, n), (n, 1, n, n), (1, n, 1, n, n)]
    generate_binary_op_test_data(
        "test_binary_op_ndarray", args.output_dir, torch.fmax, a_shapes, b_shapes
    )

    # test_add_ndarray_ndarray
    n = 10
    a_shapes = [(n, n), (n, n), (n, n), (n, n, 1), (1, n, n, n), (n, 1, n, n, n)]
    b_shapes = [(n, n), (1, n), (n, 1), (1, n, n), (n, 1, n, n), (1, n, 1, n, n)]
    generate_binary_op_test_data(
        "test_add_ndarray_ndarray",
        args.output_dir,
        lambda x, y: x + y,
        a_shapes,
        b_shapes,
    )

    # test_subtract_ndarray_ndarray
    n = 10
    a_shapes = [(n, n), (n, n), (n, n), (n, n, 1), (1, n, n, n), (n, 1, n, n, n)]
    b_shapes = [(n, n), (1, n), (n, 1), (1, n, n), (n, 1, n, n), (1, n, 1, n, n)]
    generate_binary_op_test_data(
        "test_subtract_ndarray_ndarray",
        args.output_dir,
        lambda x, y: x - y,
        a_shapes,
        b_shapes,
    )

    # test_multiply_ndarray_ndarray
    n = 10
    a_shapes = [(n, n), (n, n), (n, n), (n, n, 1), (1, n, n, n), (n, 1, n, n, n)]
    b_shapes = [(n, n), (1, n), (n, 1), (1, n, n), (n, 1, n, n), (1, n, 1, n, n)]
    generate_binary_op_test_data(
        "test_multiply_ndarray_ndarray",
        args.output_dir,
        lambda x, y: x * y,
        a_shapes,
        b_shapes,
    )

    # test_divide_ndarray_ndarray
    n = 10
    a_shapes = [(n, n), (n, n), (n, n), (n, n, 1), (1, n, n, n), (n, 1, n, n, n)]
    b_shapes = [(n, n), (1, n), (n, 1), (1, n, n), (n, 1, n, n), (1, n, 1, n, n)]
    generate_binary_op_test_data(
        "test_divide_ndarray_ndarray",
        args.output_dir,
        lambda x, y: x / y,
        a_shapes,
        b_shapes,
    )

    # test_power_ndarray_ndarray
    n = 10
    a_shapes = [(n, n), (n, n), (n, n), (n, n, 1), (1, n, n, n), (n, 1, n, n, n)]
    b_shapes = [(n, n), (1, n), (n, 1), (1, n, n), (n, 1, n, n), (1, n, 1, n, n)]
    generate_binary_op_test_data(
        "test_power_ndarray_ndarray",
        args.output_dir,
        lambda x, y: x**y,
        a_shapes,
        b_shapes,
    )

    # test_matmul_ndarray
    n = 10
    a_shapes = [
        (2, n, n // 2),
        (1, n, 1),
        (2, 1, n),
        (2, n // 2, n),
    ]
    b_shapes = [(n // 2, n), (1, 1, n), (2, n, 1), (2, n, n // 2)]
    generate_binary_op_test_data(
        "test_matmul_ndarray", args.output_dir, lambda x, y: x @ y, a_shapes, b_shapes
    )

    # test_transpose_ndarray
    n = 10
    x_shape = (2, n, n // 2)
    generate_transpose_test_data("test_transpose_ndarray", args.output_dir, x_shape)

    # test_max_ndarray
    n = 10
    x_shape = (2, n, n // 2)
    generate_reduce_test_data("test_max_ndarray", args.output_dir, x_shape, "max")

    # test_min_ndarray
    n = 10
    x_shape = (2, n, n // 2)
    generate_reduce_test_data("test_min_ndarray", args.output_dir, x_shape, "min")

    # test_sum_ndarray
    n = 10
    x_shape = (2, n, n // 2)
    generate_reduce_test_data("test_sum_ndarray", args.output_dir, x_shape, "sum")


if __name__ == "__main__":
    main()
