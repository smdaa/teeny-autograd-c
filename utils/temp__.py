import torch
from save_ndarray import save_ndarray_to_file

torch.manual_seed(0)

prefix = "test_backward"
n = 1000

a = torch.rand(n, requires_grad=True) 
b = torch.rand(n, requires_grad=True) 

c = b + (a * (a + b))

up_stream_grad = torch.rand(n, requires_grad=True)
c.backward(up_stream_grad)

save_ndarray_to_file(f"../test/test_data/{prefix}_a_val.txt", a.detach().numpy())
save_ndarray_to_file(f"../test/test_data/{prefix}_b_val.txt", b.detach().numpy())
save_ndarray_to_file(f"../test/test_data/{prefix}_c_val.txt", c.detach().numpy())
save_ndarray_to_file(f"../test/test_data/{prefix}_up_stream_grad.txt", up_stream_grad.detach().numpy())

save_ndarray_to_file(
    f"../test/test_data/{prefix}_a_grad.txt", a.grad.detach().numpy()
)
save_ndarray_to_file(
    f"../test/test_data/{prefix}_b_grad.txt", b.grad.detach().numpy()
)
