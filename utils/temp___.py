import torch
from save_ndarray import save_ndarray_to_file

torch.manual_seed(0)

prefix = "test_forward_multilayer_perceptron"

mlp = torch.nn.Sequential(
    torch.nn.Linear(32, 128, bias=True),
    torch.nn.Linear(128, 256, bias=True),
    torch.nn.Linear(256, 512, bias=True),
)

input = torch.rand(64, 32)
output = mlp(input)
up_stream_grad = torch.rand(64, 512, requires_grad=True)
output.backward(up_stream_grad)

save_ndarray_to_file(
    f"../test/test_data/{prefix}_input_val.txt", input.detach().numpy()
)
save_ndarray_to_file(
    f"../test/test_data/{prefix}_output_val.txt", output.detach().numpy()
)

save_ndarray_to_file(
    f"../test/test_data/{prefix}_weights0_val.txt", mlp[0].weight.detach().numpy().T
)
save_ndarray_to_file(
    f"../test/test_data/{prefix}_weights1_val.txt", mlp[1].weight.detach().numpy().T
)
save_ndarray_to_file(
    f"../test/test_data/{prefix}_weights2_val.txt", mlp[2].weight.detach().numpy().T
)

save_ndarray_to_file(
    f"../test/test_data/{prefix}_bias0_val.txt",
    mlp[0].bias.detach().numpy().reshape(1, -1),
)
save_ndarray_to_file(
    f"../test/test_data/{prefix}_bias1_val.txt",
    mlp[1].bias.detach().numpy().reshape(1, -1),
)
save_ndarray_to_file(
    f"../test/test_data/{prefix}_bias2_val.txt",
    mlp[2].bias.detach().numpy().reshape(1, -1),
)

save_ndarray_to_file(
    f"../test/test_data/{prefix}_up_stream_grad.txt", up_stream_grad.detach().numpy()
)

save_ndarray_to_file(
    f"../test/test_data/{prefix}_weights0_grad.txt", mlp[0].weight.grad.detach().numpy().T
)

save_ndarray_to_file(
    f"../test/test_data/{prefix}_weights1_grad.txt", mlp[1].weight.grad.detach().numpy().T
)

save_ndarray_to_file(
    f"../test/test_data/{prefix}_weights2_grad.txt", mlp[2].weight.grad.detach().numpy().T
)

save_ndarray_to_file(
    f"../test/test_data/{prefix}_bias0_grad.txt", mlp[0].bias.grad.detach().numpy().reshape(1, -1)
)

save_ndarray_to_file(
    f"../test/test_data/{prefix}_bias1_grad.txt", mlp[1].bias.grad.detach().numpy().reshape(1, -1)
)

save_ndarray_to_file(
    f"../test/test_data/{prefix}_bias2_grad.txt", mlp[2].bias.grad.detach().numpy().reshape(1, -1)
)
