import numpy as np

# Maintain reproducibility using a shared RNG
seed = 10
rng = np.random.default_rng(seed=seed)


def setSeed(newSeed):
    global seed, rng
    seed = newSeed
    rng = np.random.default_rng(seed=seed)


class Dropout:
    def __init__(self, dropOutRate):
        self.dropOutRate = dropOutRate

    def forward(self, x):
        self.mask = rng.choice(
            [0, 1], size=x.shape,
            p=[self.dropOutRate, 1 - self.dropOutRate]
        )
        return x * self.mask / (1 - self.dropOutRate)

    def backward(self, gradient):
        # Scale gradients for surviving activations
        return gradient * (self.mask / (1 - self.dropOutRate))


class Linear:
    def __init__(self, inputSize, outputSize):
        # Xavier/Glorot uniform initialization
        self.limit = np.sqrt(6 / (inputSize + outputSize))
        self.W = rng.uniform(-self.limit, self.limit, size=(outputSize, inputSize))
        self.b = np.zeros((outputSize, 1))

    def forward(self, x):
        self.x = x
        return self.W @ x + self.b

    def backward(self, gradient):
        # Average weight and bias gradients across the batch
        self.dW = (gradient @ self.x.T) / gradient.shape[1]
        self.db = np.mean(gradient, axis=1, keepdims=True)
        return self.W.T @ gradient


class ReLU:
    def forward(self, x):
        self.x = x
        return np.maximum(0, x)

    def backward(self, gradient):
        return gradient * (self.x > 0)


class Softmax:
    @staticmethod
    def forward(x):
        # Shift logits before exponentiation for numerical stability
        x = x - np.max(x, axis=0, keepdims=True)
        expX = np.exp(x)
        return expX / np.sum(expX, axis=0, keepdims=True)
