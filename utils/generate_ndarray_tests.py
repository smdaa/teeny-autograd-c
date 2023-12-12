import os
import torch
from save_ndarray import save_ndarray_to_file

torch.manual_seed(0)

test = "test_read_ndarray"
dir = f"../test/test_data/test_ndarray/{test}/"
n = 100
x = torch.eye(n, dtype=torch.double) + 1
save_ndarray_to_file(os.path.join(dir, "x.txt"), x.detach().numpy())

test = "test_unary_op_ndarray"
dir = f"../test/test_data/test_ndarray/{test}/"
n = 10
x = torch.rand(n, n, dtype=torch.double)
y = torch.sin(x)
save_ndarray_to_file(os.path.join(dir, "x.txt"), x.detach().numpy())
save_ndarray_to_file(os.path.join(dir, "y.txt"), y.detach().numpy())

test = "test_log_ndarray"
dir = f"../test/test_data/test_ndarray/{test}/"
n = 10
x = torch.rand(n, n, dtype=torch.double)
y = torch.log(x)
save_ndarray_to_file(os.path.join(dir, "x.txt"), x.detach().numpy())
save_ndarray_to_file(os.path.join(dir, "y.txt"), y.detach().numpy())

test = "test_add_ndarray_scalar"
dir = f"../test/test_data/test_ndarray/{test}/"
n = 10
x = torch.rand(n, n, dtype=torch.double)
y = x + 10
save_ndarray_to_file(os.path.join(dir, "x.txt"), x.detach().numpy())
save_ndarray_to_file(os.path.join(dir, "y.txt"), y.detach().numpy())

test = "test_subtract_ndarray_scalar"
dir = f"../test/test_data/test_ndarray/{test}/"
n = 10
x = torch.rand(n, n, dtype=torch.double)
y = x - 10
save_ndarray_to_file(os.path.join(dir, "x.txt"), x.detach().numpy())
save_ndarray_to_file(os.path.join(dir, "y.txt"), y.detach().numpy())

test = "test_multiply_ndarray_scalar"
dir = f"../test/test_data/test_ndarray/{test}/"
n = 10
x = torch.rand(n, n, dtype=torch.double)
y = x * 10
save_ndarray_to_file(os.path.join(dir, "x.txt"), x.detach().numpy())
save_ndarray_to_file(os.path.join(dir, "y.txt"), y.detach().numpy())

test = "test_divide_ndarray_scalar"
dir = f"../test/test_data/test_ndarray/{test}/"
n = 10
x = torch.rand(n, n, dtype=torch.double)
y = x / 10
save_ndarray_to_file(os.path.join(dir, "x.txt"), x.detach().numpy())
save_ndarray_to_file(os.path.join(dir, "y.txt"), y.detach().numpy())

test = "test_divide_scalar_ndarray"
dir = f"../test/test_data/test_ndarray/{test}/"
n = 10
x = torch.rand(n, n, dtype=torch.double)
y = 10 / x
save_ndarray_to_file(os.path.join(dir, "x.txt"), x.detach().numpy())
save_ndarray_to_file(os.path.join(dir, "y.txt"), y.detach().numpy())

test = "test_power_ndarray_scalar"
dir = f"../test/test_data/test_ndarray/{test}/"
n = 10
x = torch.rand(n, n, dtype=torch.double)
y = x**2
save_ndarray_to_file(os.path.join(dir, "x.txt"), x.detach().numpy())
save_ndarray_to_file(os.path.join(dir, "y.txt"), y.detach().numpy())

test = "test_binary_op_ndarray"
dir = f"../test/test_data/test_ndarray/{test}/"
n = 10
a0 = torch.rand(n, n, dtype=torch.double)
b0 = torch.rand(n, n, dtype=torch.double)
c0 = torch.fmax(a0, b0)
save_ndarray_to_file(os.path.join(dir, "a0.txt"), a0.detach().numpy())
save_ndarray_to_file(os.path.join(dir, "b0.txt"), b0.detach().numpy())
save_ndarray_to_file(os.path.join(dir, "c0.txt"), c0.detach().numpy())
a1 = torch.rand(n, n, dtype=torch.double)
b1 = torch.rand(1, n, dtype=torch.double)
c1 = torch.fmax(a1, b1)
save_ndarray_to_file(os.path.join(dir, "a1.txt"), a1.detach().numpy())
save_ndarray_to_file(os.path.join(dir, "b1.txt"), b1.detach().numpy())
save_ndarray_to_file(os.path.join(dir, "c1.txt"), c1.detach().numpy())
a2 = torch.rand(n, n, dtype=torch.double)
b2 = torch.rand(n, 1, dtype=torch.double)
c2 = torch.fmax(a2, b2)
save_ndarray_to_file(os.path.join(dir, "a2.txt"), a2.detach().numpy())
save_ndarray_to_file(os.path.join(dir, "b2.txt"), b2.detach().numpy())
save_ndarray_to_file(os.path.join(dir, "c2.txt"), c2.detach().numpy())
a3 = torch.rand(n, n, 1, dtype=torch.double)
b3 = torch.rand(1, n, n, dtype=torch.double)
c3 = torch.fmax(a3, b3)
save_ndarray_to_file(os.path.join(dir, "a3.txt"), a3.detach().numpy())
save_ndarray_to_file(os.path.join(dir, "b3.txt"), b3.detach().numpy())
save_ndarray_to_file(os.path.join(dir, "c3.txt"), c3.detach().numpy())
a4 = torch.rand(1, n, n, n, dtype=torch.double)
b4 = torch.rand(n, 1, n, n, dtype=torch.double)
c4 = torch.fmax(a4, b4)
save_ndarray_to_file(os.path.join(dir, "a4.txt"), a4.detach().numpy())
save_ndarray_to_file(os.path.join(dir, "b4.txt"), b4.detach().numpy())
save_ndarray_to_file(os.path.join(dir, "c4.txt"), c4.detach().numpy())
a5 = torch.rand(n, 1, n, n, n, dtype=torch.double)
b5 = torch.rand(1, n, 1, n, n, dtype=torch.double)
c5 = torch.fmax(a5, b5)
save_ndarray_to_file(os.path.join(dir, "a5.txt"), a5.detach().numpy())
save_ndarray_to_file(os.path.join(dir, "b5.txt"), b5.detach().numpy())
save_ndarray_to_file(os.path.join(dir, "c5.txt"), c5.detach().numpy())

test = "test_add_ndarray_ndarray"
dir = f"../test/test_data/test_ndarray/{test}/"
n = 10
a0 = torch.rand(n, n, dtype=torch.double)
b0 = torch.rand(n, n, dtype=torch.double)
c0 = a0 + b0
save_ndarray_to_file(os.path.join(dir, "a0.txt"), a0.detach().numpy())
save_ndarray_to_file(os.path.join(dir, "b0.txt"), b0.detach().numpy())
save_ndarray_to_file(os.path.join(dir, "c0.txt"), c0.detach().numpy())
a1 = torch.rand(n, n, dtype=torch.double)
b1 = torch.rand(1, n, dtype=torch.double)
c1 = a1 + b1
save_ndarray_to_file(os.path.join(dir, "a1.txt"), a1.detach().numpy())
save_ndarray_to_file(os.path.join(dir, "b1.txt"), b1.detach().numpy())
save_ndarray_to_file(os.path.join(dir, "c1.txt"), c1.detach().numpy())
a2 = torch.rand(n, n, dtype=torch.double)
b2 = torch.rand(n, 1, dtype=torch.double)
c2 = a2 + b2
save_ndarray_to_file(os.path.join(dir, "a2.txt"), a2.detach().numpy())
save_ndarray_to_file(os.path.join(dir, "b2.txt"), b2.detach().numpy())
save_ndarray_to_file(os.path.join(dir, "c2.txt"), c2.detach().numpy())
a3 = torch.rand(n, n, 1, dtype=torch.double)
b3 = torch.rand(1, n, n, dtype=torch.double)
c3 = a3 + b3
save_ndarray_to_file(os.path.join(dir, "a3.txt"), a3.detach().numpy())
save_ndarray_to_file(os.path.join(dir, "b3.txt"), b3.detach().numpy())
save_ndarray_to_file(os.path.join(dir, "c3.txt"), c3.detach().numpy())
a4 = torch.rand(1, n, n, n, dtype=torch.double)
b4 = torch.rand(n, 1, n, n, dtype=torch.double)
c4 = a4 + b4
save_ndarray_to_file(os.path.join(dir, "a4.txt"), a4.detach().numpy())
save_ndarray_to_file(os.path.join(dir, "b4.txt"), b4.detach().numpy())
save_ndarray_to_file(os.path.join(dir, "c4.txt"), c4.detach().numpy())
a5 = torch.rand(n, 1, n, n, n, dtype=torch.double)
b5 = torch.rand(1, n, 1, n, n, dtype=torch.double)
c5 = a5 + b5
save_ndarray_to_file(os.path.join(dir, "a5.txt"), a5.detach().numpy())
save_ndarray_to_file(os.path.join(dir, "b5.txt"), b5.detach().numpy())
save_ndarray_to_file(os.path.join(dir, "c5.txt"), c5.detach().numpy())

test = "test_subtract_ndarray_ndarray"
dir = f"../test/test_data/test_ndarray/{test}/"
n = 10
a0 = torch.rand(n, n, dtype=torch.double)
b0 = torch.rand(n, n, dtype=torch.double)
c0 = a0 - b0
save_ndarray_to_file(os.path.join(dir, "a0.txt"), a0.detach().numpy())
save_ndarray_to_file(os.path.join(dir, "b0.txt"), b0.detach().numpy())
save_ndarray_to_file(os.path.join(dir, "c0.txt"), c0.detach().numpy())
a1 = torch.rand(n, n, dtype=torch.double)
b1 = torch.rand(1, n, dtype=torch.double)
c1 = a1 - b1
save_ndarray_to_file(os.path.join(dir, "a1.txt"), a1.detach().numpy())
save_ndarray_to_file(os.path.join(dir, "b1.txt"), b1.detach().numpy())
save_ndarray_to_file(os.path.join(dir, "c1.txt"), c1.detach().numpy())
a2 = torch.rand(n, n, dtype=torch.double)
b2 = torch.rand(n, 1, dtype=torch.double)
c2 = a2 - b2
save_ndarray_to_file(os.path.join(dir, "a2.txt"), a2.detach().numpy())
save_ndarray_to_file(os.path.join(dir, "b2.txt"), b2.detach().numpy())
save_ndarray_to_file(os.path.join(dir, "c2.txt"), c2.detach().numpy())
a3 = torch.rand(n, n, 1, dtype=torch.double)
b3 = torch.rand(1, n, n, dtype=torch.double)
c3 = a3 - b3
save_ndarray_to_file(os.path.join(dir, "a3.txt"), a3.detach().numpy())
save_ndarray_to_file(os.path.join(dir, "b3.txt"), b3.detach().numpy())
save_ndarray_to_file(os.path.join(dir, "c3.txt"), c3.detach().numpy())
a4 = torch.rand(1, n, n, n, dtype=torch.double)
b4 = torch.rand(n, 1, n, n, dtype=torch.double)
c4 = a4 - b4
save_ndarray_to_file(os.path.join(dir, "a4.txt"), a4.detach().numpy())
save_ndarray_to_file(os.path.join(dir, "b4.txt"), b4.detach().numpy())
save_ndarray_to_file(os.path.join(dir, "c4.txt"), c4.detach().numpy())
a5 = torch.rand(n, 1, n, n, n, dtype=torch.double)
b5 = torch.rand(1, n, 1, n, n, dtype=torch.double)
c5 = a5 - b5
save_ndarray_to_file(os.path.join(dir, "a5.txt"), a5.detach().numpy())
save_ndarray_to_file(os.path.join(dir, "b5.txt"), b5.detach().numpy())
save_ndarray_to_file(os.path.join(dir, "c5.txt"), c5.detach().numpy())

test = "test_multiply_ndarray_ndarray"
dir = f"../test/test_data/test_ndarray/{test}/"
n = 10
a0 = torch.rand(n, n, dtype=torch.double)
b0 = torch.rand(n, n, dtype=torch.double)
c0 = a0 * b0
save_ndarray_to_file(os.path.join(dir, "a0.txt"), a0.detach().numpy())
save_ndarray_to_file(os.path.join(dir, "b0.txt"), b0.detach().numpy())
save_ndarray_to_file(os.path.join(dir, "c0.txt"), c0.detach().numpy())
a1 = torch.rand(n, n, dtype=torch.double)
b1 = torch.rand(1, n, dtype=torch.double)
c1 = a1 * b1
save_ndarray_to_file(os.path.join(dir, "a1.txt"), a1.detach().numpy())
save_ndarray_to_file(os.path.join(dir, "b1.txt"), b1.detach().numpy())
save_ndarray_to_file(os.path.join(dir, "c1.txt"), c1.detach().numpy())
a2 = torch.rand(n, n, dtype=torch.double)
b2 = torch.rand(n, 1, dtype=torch.double)
c2 = a2 * b2
save_ndarray_to_file(os.path.join(dir, "a2.txt"), a2.detach().numpy())
save_ndarray_to_file(os.path.join(dir, "b2.txt"), b2.detach().numpy())
save_ndarray_to_file(os.path.join(dir, "c2.txt"), c2.detach().numpy())
a3 = torch.rand(n, n, 1, dtype=torch.double)
b3 = torch.rand(1, n, n, dtype=torch.double)
c3 = a3 * b3
save_ndarray_to_file(os.path.join(dir, "a3.txt"), a3.detach().numpy())
save_ndarray_to_file(os.path.join(dir, "b3.txt"), b3.detach().numpy())
save_ndarray_to_file(os.path.join(dir, "c3.txt"), c3.detach().numpy())
a4 = torch.rand(1, n, n, n, dtype=torch.double)
b4 = torch.rand(n, 1, n, n, dtype=torch.double)
c4 = a4 * b4
save_ndarray_to_file(os.path.join(dir, "a4.txt"), a4.detach().numpy())
save_ndarray_to_file(os.path.join(dir, "b4.txt"), b4.detach().numpy())
save_ndarray_to_file(os.path.join(dir, "c4.txt"), c4.detach().numpy())
a5 = torch.rand(n, 1, n, n, n, dtype=torch.double)
b5 = torch.rand(1, n, 1, n, n, dtype=torch.double)
c5 = a5 * b5
save_ndarray_to_file(os.path.join(dir, "a5.txt"), a5.detach().numpy())
save_ndarray_to_file(os.path.join(dir, "b5.txt"), b5.detach().numpy())
save_ndarray_to_file(os.path.join(dir, "c5.txt"), c5.detach().numpy())

test = "test_divide_ndarray_ndarray"
dir = f"../test/test_data/test_ndarray/{test}/"
n = 10
a0 = torch.rand(n, n, dtype=torch.double)
b0 = torch.rand(n, n, dtype=torch.double)
c0 = a0 / b0
save_ndarray_to_file(os.path.join(dir, "a0.txt"), a0.detach().numpy())
save_ndarray_to_file(os.path.join(dir, "b0.txt"), b0.detach().numpy())
save_ndarray_to_file(os.path.join(dir, "c0.txt"), c0.detach().numpy())
a1 = torch.rand(n, n, dtype=torch.double)
b1 = torch.rand(1, n, dtype=torch.double)
c1 = a1 / b1
save_ndarray_to_file(os.path.join(dir, "a1.txt"), a1.detach().numpy())
save_ndarray_to_file(os.path.join(dir, "b1.txt"), b1.detach().numpy())
save_ndarray_to_file(os.path.join(dir, "c1.txt"), c1.detach().numpy())
a2 = torch.rand(n, n, dtype=torch.double)
b2 = torch.rand(n, 1, dtype=torch.double)
c2 = a2 / b2
save_ndarray_to_file(os.path.join(dir, "a2.txt"), a2.detach().numpy())
save_ndarray_to_file(os.path.join(dir, "b2.txt"), b2.detach().numpy())
save_ndarray_to_file(os.path.join(dir, "c2.txt"), c2.detach().numpy())
a3 = torch.rand(n, n, 1, dtype=torch.double)
b3 = torch.rand(1, n, n, dtype=torch.double)
c3 = a3 / b3
save_ndarray_to_file(os.path.join(dir, "a3.txt"), a3.detach().numpy())
save_ndarray_to_file(os.path.join(dir, "b3.txt"), b3.detach().numpy())
save_ndarray_to_file(os.path.join(dir, "c3.txt"), c3.detach().numpy())
a4 = torch.rand(1, n, n, n, dtype=torch.double)
b4 = torch.rand(n, 1, n, n, dtype=torch.double)
c4 = a4 / b4
save_ndarray_to_file(os.path.join(dir, "a4.txt"), a4.detach().numpy())
save_ndarray_to_file(os.path.join(dir, "b4.txt"), b4.detach().numpy())
save_ndarray_to_file(os.path.join(dir, "c4.txt"), c4.detach().numpy())
a5 = torch.rand(n, 1, n, n, n, dtype=torch.double)
b5 = torch.rand(1, n, 1, n, n, dtype=torch.double)
c5 = a5 / b5
save_ndarray_to_file(os.path.join(dir, "a5.txt"), a5.detach().numpy())
save_ndarray_to_file(os.path.join(dir, "b5.txt"), b5.detach().numpy())
save_ndarray_to_file(os.path.join(dir, "c5.txt"), c5.detach().numpy())

test = "test_power_ndarray_ndarray"
dir = f"../test/test_data/test_ndarray/{test}/"
n = 10
a0 = torch.rand(n, n, dtype=torch.double)
b0 = torch.rand(n, n, dtype=torch.double)
c0 = a0**b0
save_ndarray_to_file(os.path.join(dir, "a0.txt"), a0.detach().numpy())
save_ndarray_to_file(os.path.join(dir, "b0.txt"), b0.detach().numpy())
save_ndarray_to_file(os.path.join(dir, "c0.txt"), c0.detach().numpy())
a1 = torch.rand(n, n, dtype=torch.double)
b1 = torch.rand(1, n, dtype=torch.double)
c1 = a1**b1
save_ndarray_to_file(os.path.join(dir, "a1.txt"), a1.detach().numpy())
save_ndarray_to_file(os.path.join(dir, "b1.txt"), b1.detach().numpy())
save_ndarray_to_file(os.path.join(dir, "c1.txt"), c1.detach().numpy())
a2 = torch.rand(n, n, dtype=torch.double)
b2 = torch.rand(n, 1, dtype=torch.double)
c2 = a2**b2
save_ndarray_to_file(os.path.join(dir, "a2.txt"), a2.detach().numpy())
save_ndarray_to_file(os.path.join(dir, "b2.txt"), b2.detach().numpy())
save_ndarray_to_file(os.path.join(dir, "c2.txt"), c2.detach().numpy())
a3 = torch.rand(n, n, 1, dtype=torch.double)
b3 = torch.rand(1, n, n, dtype=torch.double)
c3 = a3**b3
save_ndarray_to_file(os.path.join(dir, "a3.txt"), a3.detach().numpy())
save_ndarray_to_file(os.path.join(dir, "b3.txt"), b3.detach().numpy())
save_ndarray_to_file(os.path.join(dir, "c3.txt"), c3.detach().numpy())
a4 = torch.rand(1, n, n, n, dtype=torch.double)
b4 = torch.rand(n, 1, n, n, dtype=torch.double)
c4 = a4**b4
save_ndarray_to_file(os.path.join(dir, "a4.txt"), a4.detach().numpy())
save_ndarray_to_file(os.path.join(dir, "b4.txt"), b4.detach().numpy())
save_ndarray_to_file(os.path.join(dir, "c4.txt"), c4.detach().numpy())
a5 = torch.rand(n, 1, n, n, n, dtype=torch.double)
b5 = torch.rand(1, n, 1, n, n, dtype=torch.double)
c5 = a5**b5
save_ndarray_to_file(os.path.join(dir, "a5.txt"), a5.detach().numpy())
save_ndarray_to_file(os.path.join(dir, "b5.txt"), b5.detach().numpy())
save_ndarray_to_file(os.path.join(dir, "c5.txt"), c5.detach().numpy())

test = "test_matmul_ndarray"
dir = f"../test/test_data/test_ndarray/{test}/"
n = 10
a0 = torch.rand(2, n, n // 2, dtype=torch.double)
b0 = torch.rand(n // 2, n, dtype=torch.double)
c0 = a0 @ b0
save_ndarray_to_file(os.path.join(dir, "a0.txt"), a0.detach().numpy())
save_ndarray_to_file(os.path.join(dir, "b0.txt"), b0.detach().numpy())
save_ndarray_to_file(os.path.join(dir, "c0.txt"), c0.detach().numpy())
a1 = torch.rand(1, n, 1, dtype=torch.double)
b1 = torch.rand(1, 1, n, dtype=torch.double)
c1 = a1 @ b1
save_ndarray_to_file(os.path.join(dir, "a1.txt"), a1.detach().numpy())
save_ndarray_to_file(os.path.join(dir, "b1.txt"), b1.detach().numpy())
save_ndarray_to_file(os.path.join(dir, "c1.txt"), c1.detach().numpy())
a2 = torch.rand(2, 1, n, dtype=torch.double)
b2 = torch.rand(2, n, 1, dtype=torch.double)
c2 = a2 @ b2
save_ndarray_to_file(os.path.join(dir, "a2.txt"), a2.detach().numpy())
save_ndarray_to_file(os.path.join(dir, "b2.txt"), b2.detach().numpy())
save_ndarray_to_file(os.path.join(dir, "c2.txt"), c2.detach().numpy())
a3 = torch.rand(2, n // 2, n, dtype=torch.double)
b3 = torch.rand(2, n, n // 2, dtype=torch.double)
c3 = a3 @ b3
save_ndarray_to_file(os.path.join(dir, "a3.txt"), a3.detach().numpy())
save_ndarray_to_file(os.path.join(dir, "b3.txt"), b3.detach().numpy())
save_ndarray_to_file(os.path.join(dir, "c3.txt"), c3.detach().numpy())


test = "test_transpose_ndarray"
dir = f"../test/test_data/test_ndarray/{test}/"
n = 10
x = torch.rand(2, n, n // 2, dtype=torch.double)
y0 = x.transpose(0, 1)
y1 = x.transpose(0, 2)
y2 = x.transpose(1, 2)
save_ndarray_to_file(os.path.join(dir, "x.txt"), x.detach().numpy())
save_ndarray_to_file(os.path.join(dir, "y0.txt"), y0.detach().numpy())
save_ndarray_to_file(os.path.join(dir, "y1.txt"), y1.detach().numpy())
save_ndarray_to_file(os.path.join(dir, "y2.txt"), y2.detach().numpy())

test = "test_max_ndarray"
dir = f"../test/test_data/test_ndarray/{test}/"
n = 10
x = torch.rand(2, n, n // 2, dtype=torch.double)
y0, _ = x.max(0, keepdim=True)
y1, _ = x.max(1, keepdim=True)
y2, _ = x.max(2, keepdim=True)
save_ndarray_to_file(os.path.join(dir, "x.txt"), x.detach().numpy())
save_ndarray_to_file(os.path.join(dir, "y0.txt"), y0.detach().numpy())
save_ndarray_to_file(os.path.join(dir, "y1.txt"), y1.detach().numpy())
save_ndarray_to_file(os.path.join(dir, "y2.txt"), y2.detach().numpy())

test = "test_min_ndarray"
dir = f"../test/test_data/test_ndarray/{test}/"
n = 10
x = torch.rand(2, n, n // 2, dtype=torch.double)
y0, _ = x.min(0, keepdim=True)
y1, _ = x.min(1, keepdim=True)
y2, _ = x.min(2, keepdim=True)
save_ndarray_to_file(os.path.join(dir, "x.txt"), x.detach().numpy())
save_ndarray_to_file(os.path.join(dir, "y0.txt"), y0.detach().numpy())
save_ndarray_to_file(os.path.join(dir, "y1.txt"), y1.detach().numpy())
save_ndarray_to_file(os.path.join(dir, "y2.txt"), y2.detach().numpy())

test = "test_sum_ndarray"
dir = f"../test/test_data/test_ndarray/{test}/"
n = 10
x = torch.rand(2, n, n // 2, dtype=torch.double)
y0 = x.sum(0, keepdim=True)
y1 = x.sum(1, keepdim=True)
y2 = x.sum(2, keepdim=True)
save_ndarray_to_file(os.path.join(dir, "x.txt"), x.detach().numpy())
save_ndarray_to_file(os.path.join(dir, "y0.txt"), y0.detach().numpy())
save_ndarray_to_file(os.path.join(dir, "y1.txt"), y1.detach().numpy())
save_ndarray_to_file(os.path.join(dir, "y2.txt"), y2.detach().numpy())
