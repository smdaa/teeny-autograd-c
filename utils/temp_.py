import torch
from save_ndarray import save_ndarray_to_file

torch.manual_seed(0)

prefix = "test_forward_linear_layer"

in_size = 128
out_size = 256

layer = torch.nn.Linear(in_size, out_size, bias=True)
input = torch.randn(1, in_size)
output = layer(input)


save_ndarray_to_file(f"../test/test_data/{prefix}_input_val.txt", input.detach().numpy())
save_ndarray_to_file(f"../test/test_data/{prefix}_weights_val.txt", layer.weight.detach().numpy().T)
save_ndarray_to_file(f"../test/test_data/{prefix}_bias_val.txt", layer.bias.detach().numpy().reshape(1, -1))
save_ndarray_to_file(f"../test/test_data/{prefix}_output_val.txt", output.detach().numpy())

