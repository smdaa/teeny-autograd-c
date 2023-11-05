import torch
from save_ndarray import save_ndarray_to_file

torch.manual_seed(0)

prefix = "test_power_variable"
n = 1000

var1 = torch.rand(n, requires_grad=True) + 1
var2 = torch.rand(n, requires_grad=True) + 1
var = var1 ** var2

var1.retain_grad()
var2.retain_grad()

up_stream_grad = torch.rand(n, requires_grad=True)
var.backward(up_stream_grad)

save_ndarray_to_file(f"../test/test_data/{prefix}_var1_val.txt", var1.detach().numpy())
save_ndarray_to_file(f"../test/test_data/{prefix}_var2_val.txt", var2.detach().numpy())
save_ndarray_to_file(f"../test/test_data/{prefix}_var_val.txt", var.detach().numpy())
save_ndarray_to_file(
    f"../test/test_data/{prefix}_up_stream_grad.txt", up_stream_grad.detach().numpy()
)
save_ndarray_to_file(
    f"../test/test_data/{prefix}_var1_grad.txt", var1.grad.detach().numpy()
)
save_ndarray_to_file(
    f"../test/test_data/{prefix}_var2_grad.txt", var2.grad.detach().numpy()
)
