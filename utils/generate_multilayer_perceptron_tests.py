import os
import torch
from save_ndarray import save_ndarray_to_file

torch.manual_seed(0)

test = "test_forward_multilayer_perceptron"
dir = f"../test/test_data/test_multilayer_perceptron/{test}"
mlp = torch.nn.Sequential(
    torch.nn.Linear(32, 128, bias=True, dtype=torch.double),
    torch.nn.Linear(128, 256, bias=True, dtype=torch.double),
    torch.nn.Linear(256, 512, bias=True, dtype=torch.double),
)
x = torch.rand(64, 32, dtype=torch.double, requires_grad=True)
y = mlp(x)
z = torch.rand(64, 512, dtype=torch.double, requires_grad=True)
y.backward(z)
save_ndarray_to_file(os.path.join(dir, "x.txt"), x.detach().numpy())
save_ndarray_to_file(os.path.join(dir, "y.txt"), y.detach().numpy())
save_ndarray_to_file(
    os.path.join(dir, "weights0.txt"), mlp[0].weight.detach().numpy().T
)
save_ndarray_to_file(
    os.path.join(dir, "weights1.txt"), mlp[1].weight.detach().numpy().T
)
save_ndarray_to_file(
    os.path.join(dir, "weights2.txt"), mlp[2].weight.detach().numpy().T
)
save_ndarray_to_file(
    os.path.join(
        dir,
        "bias0.txt",
    ),
    mlp[0].bias.detach().numpy().reshape(1, -1),
)
save_ndarray_to_file(
    os.path.join(dir, "bias1.txt"),
    mlp[1].bias.detach().numpy().reshape(1, -1),
)
save_ndarray_to_file(
    os.path.join(dir, "bias2.txt"),
    mlp[2].bias.detach().numpy().reshape(1, -1),
)
save_ndarray_to_file(os.path.join(dir, "z.txt"), z.detach().numpy())
save_ndarray_to_file(
    os.path.join(dir, "weights0_grad.txt"),
    mlp[0].weight.grad.detach().numpy().T,
)
save_ndarray_to_file(
    os.path.join(dir, "weights1_grad.txt"),
    mlp[1].weight.grad.detach().numpy().T,
)
save_ndarray_to_file(
    os.path.join(dir, "weights2_grad.txt"),
    mlp[2].weight.grad.detach().numpy().T,
)
save_ndarray_to_file(
    os.path.join(dir, "bias0_grad.txt"),
    mlp[0].bias.grad.detach().numpy().reshape(1, -1),
)
save_ndarray_to_file(
    os.path.join(dir, "bias1_grad.txt"),
    mlp[1].bias.grad.detach().numpy().reshape(1, -1),
)
save_ndarray_to_file(
    os.path.join(dir, "bias2_grad.txt"),
    mlp[2].bias.grad.detach().numpy().reshape(1, -1),
)
