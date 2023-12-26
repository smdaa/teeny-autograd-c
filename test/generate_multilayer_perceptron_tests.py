import argparse
import os
import torch

from save_ndarray import save_ndarray_to_file


def main():
    parser = argparse.ArgumentParser(
        description="Generate test data for mlp operations"
    )
    parser.add_argument(
        "--output_dir",
        default="./test_data/test_multilayer_perceptron/",
        help="Directory path to save the .txt files",
    )
    args = parser.parse_args()
    torch.manual_seed(0)

    test = "test_forward_multilayer_perceptron"
    dir_path = os.path.join(args.output_dir, test)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    mlp = torch.nn.Sequential(
        torch.nn.Linear(32, 128, bias=True, dtype=torch.double),
        torch.nn.Linear(128, 256, bias=True, dtype=torch.double),
        torch.nn.Linear(256, 512, bias=True, dtype=torch.double),
    )
    x = torch.rand(64, 32, dtype=torch.double, requires_grad=True)
    y = mlp(x)
    z = torch.rand(64, 512, dtype=torch.double, requires_grad=True)
    y.backward(z)
    save_ndarray_to_file(os.path.join(dir_path, "x.txt"), x.detach().numpy())
    save_ndarray_to_file(os.path.join(dir_path, "y.txt"), y.detach().numpy())
    save_ndarray_to_file(
        os.path.join(dir_path, "weights0.txt"), mlp[0].weight.detach().numpy().T
    )
    save_ndarray_to_file(
        os.path.join(dir_path, "weights1.txt"), mlp[1].weight.detach().numpy().T
    )
    save_ndarray_to_file(
        os.path.join(dir_path, "weights2.txt"), mlp[2].weight.detach().numpy().T
    )
    save_ndarray_to_file(
        os.path.join(
            dir_path,
            "bias0.txt",
        ),
        mlp[0].bias.detach().numpy().reshape(1, -1),
    )
    save_ndarray_to_file(
        os.path.join(dir_path, "bias1.txt"),
        mlp[1].bias.detach().numpy().reshape(1, -1),
    )
    save_ndarray_to_file(
        os.path.join(dir_path, "bias2.txt"),
        mlp[2].bias.detach().numpy().reshape(1, -1),
    )
    save_ndarray_to_file(os.path.join(dir_path, "z.txt"), z.detach().numpy())
    save_ndarray_to_file(
        os.path.join(dir_path, "weights0_grad.txt"),
        mlp[0].weight.grad.detach().numpy().T,
    )
    save_ndarray_to_file(
        os.path.join(dir_path, "weights1_grad.txt"),
        mlp[1].weight.grad.detach().numpy().T,
    )
    save_ndarray_to_file(
        os.path.join(dir_path, "weights2_grad.txt"),
        mlp[2].weight.grad.detach().numpy().T,
    )
    save_ndarray_to_file(
        os.path.join(dir_path, "bias0_grad.txt"),
        mlp[0].bias.grad.detach().numpy().reshape(1, -1),
    )
    save_ndarray_to_file(
        os.path.join(dir_path, "bias1_grad.txt"),
        mlp[1].bias.grad.detach().numpy().reshape(1, -1),
    )
    save_ndarray_to_file(
        os.path.join(dir_path, "bias2_grad.txt"),
        mlp[2].bias.grad.detach().numpy().reshape(1, -1),
    )


if __name__ == "__main__":
    main()
