import torch
import matplotlib.pyplot as plt
from activations import PiecewiseActivation

activation = PiecewiseActivation(n_points=10, left=-5, right=5)

with torch.no_grad():
    xs = torch.linspace(-10, 10, 1000)
    ys = activation(xs)

plt.scatter(xs, ys)
plt.show()