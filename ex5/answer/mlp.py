import numpy as np


def softmax(logits):
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp_logits = np.exp(shifted)
    return exp_logits / exp_logits.sum(axis=1, keepdims=True)


def cross_entropy(probs, y):
    eps = 1e-12
    return float(-np.mean(np.log(probs[np.arange(len(y)), y] + eps)))


class MLP:
    def __init__(self, input_dim, hidden_dim, output_dim, seed=0, weight_scale=0.01):
        rng = np.random.default_rng(seed)
        self.params = {
            "W1": rng.normal(0.0, weight_scale, size=(input_dim, hidden_dim)),
            "b1": np.zeros(hidden_dim),
            "W2": rng.normal(0.0, weight_scale, size=(hidden_dim, output_dim)),
            "b2": np.zeros(output_dim),
        }

    def forward(self, X):
        W1, b1 = self.params["W1"], self.params["b1"]
        W2, b2 = self.params["W2"], self.params["b2"]
        z1 = X @ W1 + b1
        h1 = np.maximum(0.0, z1)
        logits = h1 @ W2 + b2
        probs = softmax(logits)
        cache = {"X": X, "z1": z1, "h1": h1, "probs": probs}
        return probs, cache

    def loss(self, X, y, l2=0.0):
        probs, _ = self.forward(X)
        data_loss = cross_entropy(probs, y)
        reg_loss = 0.5 * l2 * (
            np.sum(self.params["W1"] ** 2) + np.sum(self.params["W2"] ** 2)
        )
        return data_loss + reg_loss

    def gradients(self, X, y, l2=0.0):
        probs, cache = self.forward(X)
        n = X.shape[0]

        dlogits = probs.copy()
        dlogits[np.arange(n), y] -= 1.0
        dlogits /= n

        grads = {}
        grads["W2"] = cache["h1"].T @ dlogits + l2 * self.params["W2"]
        grads["b2"] = dlogits.sum(axis=0)

        dh1 = dlogits @ self.params["W2"].T
        dz1 = dh1 * (cache["z1"] > 0.0)
        grads["W1"] = cache["X"].T @ dz1 + l2 * self.params["W1"]
        grads["b1"] = dz1.sum(axis=0)

        loss = cross_entropy(probs, y) + 0.5 * l2 * (
            np.sum(self.params["W1"] ** 2) + np.sum(self.params["W2"] ** 2)
        )
        return loss, grads

    def step(self, grads, lr):
        for key in self.params:
            self.params[key] -= lr * grads[key]

    def predict(self, X):
        probs, _ = self.forward(X)
        return np.argmax(probs, axis=1)
