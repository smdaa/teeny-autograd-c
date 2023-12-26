import argparse
import os
import torch

from save_ndarray import save_ndarray_to_file


def generate_unary_op_test_data(test, output_dir, unary_op, x_shape):
    dir_path = os.path.join(output_dir, test)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    x = torch.rand(x_shape, dtype=torch.double, requires_grad=True)
    y = unary_op(x)
    z = torch.rand(x_shape, dtype=torch.double, requires_grad=True)
    y.backward(z)
    save_ndarray_to_file(os.path.join(dir_path, "x.txt"), x.detach().numpy())
    save_ndarray_to_file(os.path.join(dir_path, "y.txt"), y.detach().numpy())
    save_ndarray_to_file(os.path.join(dir_path, "z.txt"), z.detach().numpy())
    save_ndarray_to_file(os.path.join(dir_path, "x_grad.txt"), x.grad.detach().numpy())


def generate_binary_op_test_data(test, output_dir, binary_op, a_shapes, b_shapes):
    dir_path = os.path.join(output_dir, test)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    for i, (a_shape, b_shape) in enumerate(zip(a_shapes, b_shapes)):
        a = torch.rand(a_shape, dtype=torch.double, requires_grad=True)
        b = torch.rand(b_shape, dtype=torch.double, requires_grad=True)
        c = binary_op(a, b)
        d = torch.rand(c.shape, requires_grad=True, dtype=torch.double)
        c.backward(d)
        save_ndarray_to_file(os.path.join(dir_path, f"a{i}.txt"), a.detach().numpy())
        save_ndarray_to_file(os.path.join(dir_path, f"b{i}.txt"), b.detach().numpy())
        save_ndarray_to_file(os.path.join(dir_path, f"c{i}.txt"), c.detach().numpy())
        save_ndarray_to_file(os.path.join(dir_path, f"d{i}.txt"), d.detach().numpy())
        save_ndarray_to_file(
            os.path.join(dir_path, f"a{i}_grad.txt"), a.grad.detach().numpy()
        )
        save_ndarray_to_file(
            os.path.join(dir_path, f"b{i}_grad.txt"), b.grad.detach().numpy()
        )


def generate_reduce_test_data(test, output_dir, x_shape, operation_type):
    assert len(x_shape) >= 3
    dir_path = os.path.join(output_dir, test)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    if operation_type == "sum":
        x0 = torch.rand(x_shape, dtype=torch.double, requires_grad=True)
        y0 = torch.sum(x0, dim=0, keepdim=True)
        z0 = torch.rand(y0.shape, dtype=torch.double, requires_grad=True)
        y0.backward(z0)
        x1 = torch.rand(x_shape, dtype=torch.double, requires_grad=True)
        y1 = torch.sum(x1, dim=1, keepdim=True)
        z1 = torch.rand(y1.shape, dtype=torch.double, requires_grad=True)
        y1.backward(z1)
        x2 = torch.rand(x_shape, dtype=torch.double, requires_grad=True)
        y2 = torch.sum(x2, dim=2, keepdim=True)
        z2 = torch.rand(y2.shape, dtype=torch.double, requires_grad=True)
        y2.backward(z2)
    elif operation_type == "softmax":
        x0 = torch.rand(x_shape, dtype=torch.double, requires_grad=True)
        y0 = torch.softmax(x0, dim=0)
        z0 = torch.rand(y0.shape, dtype=torch.double, requires_grad=True)
        y0.backward(z0)
        x1 = torch.rand(x_shape, dtype=torch.double, requires_grad=True)
        y1 = torch.softmax(x1, dim=1)
        z1 = torch.rand(y1.shape, dtype=torch.double, requires_grad=True)
        y1.backward(z1)
        x2 = torch.rand(x_shape, dtype=torch.double, requires_grad=True)
        y2 = torch.softmax(x2, dim=2)
        z2 = torch.rand(y2.shape, dtype=torch.double, requires_grad=True)
        y2.backward(z2)
    else:
        raise ValueError("Invalid operation type")
    save_ndarray_to_file(os.path.join(dir_path, "x0.txt"), x0.detach().numpy())
    save_ndarray_to_file(os.path.join(dir_path, "y0.txt"), y0.detach().numpy())
    save_ndarray_to_file(os.path.join(dir_path, "z0.txt"), z0.detach().numpy())
    save_ndarray_to_file(os.path.join(dir_path, "x1.txt"), x1.detach().numpy())
    save_ndarray_to_file(os.path.join(dir_path, "y1.txt"), y1.detach().numpy())
    save_ndarray_to_file(os.path.join(dir_path, "z1.txt"), z1.detach().numpy())
    save_ndarray_to_file(os.path.join(dir_path, "x2.txt"), x2.detach().numpy())
    save_ndarray_to_file(os.path.join(dir_path, "y2.txt"), y2.detach().numpy())
    save_ndarray_to_file(os.path.join(dir_path, "z2.txt"), z2.detach().numpy())
    save_ndarray_to_file(
        os.path.join(dir_path, "x0_grad.txt"), x0.grad.detach().numpy()
    )
    save_ndarray_to_file(
        os.path.join(dir_path, "x1_grad.txt"), x1.grad.detach().numpy()
    )
    save_ndarray_to_file(
        os.path.join(dir_path, "x2_grad.txt"), x2.grad.detach().numpy()
    )


def main():
    parser = argparse.ArgumentParser(
        description="Generate test data for variable operations"
    )
    parser.add_argument(
        "--output_dir",
        default="./test_data/test_variable/",
        help="Directory path to save the .txt files",
    )
    args = parser.parse_args()
    torch.manual_seed(0)

    # test_add_variable
    n = 10
    a_shapes = [(n, n), (n, n), (n, n), (n, n, 1), (1, n, n, n), (n, 1, n, n, n)]
    b_shapes = [(n, n), (1, n), (n, 1), (1, n, n), (n, 1, n, n), (1, n, 1, n, n)]
    generate_binary_op_test_data(
        "test_add_variable", args.output_dir, lambda x, y: x + y, a_shapes, b_shapes
    )

    # test_subtract_variable
    n = 10
    a_shapes = [(n, n), (n, n), (n, n), (n, n, 1), (1, n, n, n), (n, 1, n, n, n)]
    b_shapes = [(n, n), (1, n), (n, 1), (1, n, n), (n, 1, n, n), (1, n, 1, n, n)]
    generate_binary_op_test_data(
        "test_subtract_variable",
        args.output_dir,
        lambda x, y: x - y,
        a_shapes,
        b_shapes,
    )

    # test_multiply_variable
    n = 10
    a_shapes = [(n, n), (n, n), (n, n), (n, n, 1), (1, n, n, n), (n, 1, n, n, n)]
    b_shapes = [(n, n), (1, n), (n, 1), (1, n, n), (n, 1, n, n), (1, n, 1, n, n)]
    generate_binary_op_test_data(
        "test_multiply_variable",
        args.output_dir,
        lambda x, y: x * y,
        a_shapes,
        b_shapes,
    )

    # test_divide_variable
    n = 10
    a_shapes = [(n, n), (n, n), (n, n), (n, n, 1), (1, n, n, n), (n, 1, n, n, n)]
    b_shapes = [(n, n), (1, n), (n, 1), (1, n, n), (n, 1, n, n), (1, n, 1, n, n)]
    generate_binary_op_test_data(
        "test_divide_variable", args.output_dir, lambda x, y: x / y, a_shapes, b_shapes
    )

    # test_power_variable
    n = 10
    a_shapes = [(n, n), (n, n), (n, n), (n, n, 1), (1, n, n, n), (n, 1, n, n, n)]
    b_shapes = [(n, n), (1, n), (n, 1), (1, n, n), (n, 1, n, n), (1, n, 1, n, n)]
    generate_binary_op_test_data(
        "test_power_variable", args.output_dir, lambda x, y: x**y, a_shapes, b_shapes
    )

    # test_exp_variable
    n = 10
    x_shape = (n, n)
    generate_unary_op_test_data(
        "test_exp_variable", args.output_dir, torch.exp, x_shape
    )

    # test_relu_variable
    n = 10
    x_shape = (n, n)
    generate_unary_op_test_data(
        "test_relu_variable", args.output_dir, torch.relu, x_shape
    )

    # test_sigmoid_variable
    n = 10
    x_shape = (n, n)
    generate_unary_op_test_data(
        "test_sigmoid_variable", args.output_dir, torch.sigmoid, x_shape
    )

    # test_tanh_variable
    n = 10
    x_shape = (n, n)
    generate_unary_op_test_data(
        "test_tanh_variable", args.output_dir, torch.tanh, x_shape
    )

    # test_sum_variable
    n = 10
    x_shape = (2, n, n // 2)
    generate_reduce_test_data("test_sum_variable", args.output_dir, x_shape, "sum")

    # test_softmax_variable
    n = 10
    x_shape = (2, n, n // 2)
    generate_reduce_test_data(
        "test_softmax_variable", args.output_dir, x_shape, "softmax"
    )

    # test_matmul_variable
    n = 10
    a_shapes = [
        (n, n),
        (1, n, 1),
        (2, 1, n),
        (2, n // 2, n),
    ]
    b_shapes = [(n, n), (1, 1, n), (2, n, 1), (2, n, n // 2)]
    generate_binary_op_test_data(
        "test_matmul_variable", args.output_dir, lambda x, y: x @ y, a_shapes, b_shapes
    )

    # test_backward_variable
    test = "test_backward_variable"
    dir_path = os.path.join(args.output_dir, test)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    n = 10
    a = torch.rand(n, n, dtype=torch.double, requires_grad=True)
    b = torch.rand(n, n, dtype=torch.double, requires_grad=True)
    c = torch.rand(n, n, dtype=torch.double, requires_grad=True)
    e = torch.rand(n, n, dtype=torch.double, requires_grad=True)
    d = a * ((a * b) + c) + c
    d.backward(e)
    save_ndarray_to_file(os.path.join(dir_path, "a.txt"), a.detach().numpy())
    save_ndarray_to_file(os.path.join(dir_path, "b.txt"), b.detach().numpy())
    save_ndarray_to_file(os.path.join(dir_path, "c.txt"), c.detach().numpy())
    save_ndarray_to_file(os.path.join(dir_path, "d.txt"), d.detach().numpy())
    save_ndarray_to_file(os.path.join(dir_path, "e.txt"), e.detach().numpy())
    save_ndarray_to_file(os.path.join(dir_path, "a_grad.txt"), a.grad.detach().numpy())
    save_ndarray_to_file(os.path.join(dir_path, "b_grad.txt"), b.grad.detach().numpy())
    save_ndarray_to_file(os.path.join(dir_path, "c_grad.txt"), c.grad.detach().numpy())


if __name__ == "__main__":
    main()
