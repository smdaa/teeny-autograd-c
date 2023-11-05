import numpy as np
import torch
from save_ndarray import save_ndarray_to_file

torch.manual_seed(0)

prefix = "test_forward_linear_layer"

in_size = 128
out_size = 256
batch_size = 32

layer = torch.nn.Linear(in_size, out_size, bias=True)
input = torch.randn(batch_size, in_size)
output = layer(input)
up_stream_grad = torch.rand(batch_size, out_size, requires_grad=True)
output.backward(up_stream_grad)

save_ndarray_to_file(
    f"../test/test_data/{prefix}_input_val.txt", input.detach().numpy()
)
save_ndarray_to_file(
    f"../test/test_data/{prefix}_weights_val.txt", layer.weight.detach().numpy().T
)
save_ndarray_to_file(
    f"../test/test_data/{prefix}_bias_val.txt",
    layer.bias.detach().numpy().reshape(1, -1),
)
save_ndarray_to_file(
    f"../test/test_data/{prefix}_output_val.txt", output.detach().numpy()
)

save_ndarray_to_file(
    f"../test/test_data/{prefix}_up_stream_grad.txt", up_stream_grad.detach().numpy()
)

save_ndarray_to_file(
    f"../test/test_data/{prefix}_weights_grad.txt", layer.weight.grad.detach().numpy().T
)

save_ndarray_to_file(
    f"../test/test_data/{prefix}_bias_grad.txt", layer.bias.grad.detach().numpy().reshape(1, -1)
)
