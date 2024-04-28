import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import time

# from components import MLP
from activations import PiecewiseActivation


training_pts = 1000
training_scale = 2.0
hidden = 200
lr = 0.01

xs = torch.rand(size=(training_pts, 1)) * training_scale * 2 - training_scale
val_xs = torch.rand(size=(training_pts, 1)) * training_scale * 2 - training_scale
ys = torch.sin(xs * 30 / training_scale)
print(xs)

plt.ion()
plt.show(block=False)


activation = PiecewiseActivation(n_points=100, left=-2, right=2)
model = activation
optimizer = torch.optim.Adam(model.parameters(), lr=lr)


def visualize():
    with torch.no_grad():
        val_y_preds = model(val_xs)
        # val_y_preds = model.activation(val_xs)

    plt.clf()
    plt.scatter(xs, ys, color='blue')
    plt.scatter(val_xs, val_y_preds, color='orange')
    plt.draw()
    plt.pause(0.01)

for epoch in range(10000):
    y_pred = model(xs)
    loss = F.mse_loss(y_pred, ys)
    print(f"epoch {epoch}: {loss}")
    optimizer.zero_grad()
    loss.backward()

    # print('y grads', model.activation.ys.grad)

    optimizer.step()

    if epoch % 10 == 0:
        visualize()
