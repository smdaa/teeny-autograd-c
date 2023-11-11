import torch
from save_ndarray import save_ndarray_to_file

torch.manual_seed(0)

prefix = "test_sigmoid_variable"
n = 1000

var = torch.rand(n, requires_grad=True)
nvar = torch.sigmoid(var)

up_stream_grad = torch.rand(n, requires_grad=True)
nvar.backward(up_stream_grad)

save_ndarray_to_file(f"../test/test_data/{prefix}_var_val.txt", var.detach().numpy())
save_ndarray_to_file(f"../test/test_data/{prefix}_nvar_val.txt", nvar.detach().numpy())
save_ndarray_to_file(
    f"../test/test_data/{prefix}_up_stream_grad.txt", up_stream_grad.detach().numpy()
)
save_ndarray_to_file(
    f"../test/test_data/{prefix}_var_grad.txt", var.grad.detach().numpy()
)

