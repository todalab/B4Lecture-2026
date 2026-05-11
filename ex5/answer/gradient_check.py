import numpy as np

from mlp import MLP


def numerical_gradient(model, X, y, param_name, index, eps, l2):
    original = model.params[param_name][index]

    model.params[param_name][index] = original + eps
    loss_plus = model.loss(X, y, l2=l2)

    model.params[param_name][index] = original - eps
    loss_minus = model.loss(X, y, l2=l2)

    model.params[param_name][index] = original
    return (loss_plus - loss_minus) / (2.0 * eps)


def main():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(5, 4))
    y = np.array([0, 1, 2, 1, 0])
    model = MLP(input_dim=4, hidden_dim=6, output_dim=3, seed=2, weight_scale=0.1)

    l2 = 1e-3
    eps = 1e-5
    loss, grads = model.gradients(X, y, l2=l2)
    print(f"loss: {loss:.8f}")

    checks = [
        ("W1", (0, 0)),
        ("W1", (2, 3)),
        ("b1", (1,)),
        ("W2", (0, 1)),
        ("W2", (4, 2)),
        ("b2", (2,)),
    ]

    max_abs_diff = 0.0
    for param_name, index in checks:
        num_grad = numerical_gradient(model, X, y, param_name, index, eps, l2)
        bp_grad = grads[param_name][index]
        abs_diff = abs(num_grad - bp_grad)
        max_abs_diff = max(max_abs_diff, abs_diff)
        print(
            f"{param_name}{index}: "
            f"backprop={bp_grad:.8e}, "
            f"numerical={num_grad:.8e}, "
            f"abs_diff={abs_diff:.3e}"
        )

    print(f"max abs diff: {max_abs_diff:.3e}")


if __name__ == "__main__":
    main()
