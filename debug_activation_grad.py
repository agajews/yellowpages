import torch
from activations import PiecewiseActivation

activation = PiecewiseActivation(n_points=10, left=-5, right=5)

x = torch.tensor(-10.0, requires_grad=True)
ys = activation(x)
ys.backward()
print(x.grad)
print(activation.ys.grad)