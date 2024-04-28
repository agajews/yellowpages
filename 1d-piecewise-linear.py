import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import time

# from components import MLP
from activations import PiecewiseActivation


training_pts = 1000
training_scale = 2.0
hidden = 200
lr = 0.0001

xs = torch.rand(size=(training_pts, 1)) * training_scale * 2 - training_scale
val_xs = torch.rand(size=(training_pts, 1)) * training_scale * 2 - training_scale
ys = torch.sin(xs * 30 / training_scale)
print(xs)

plt.ion()
plt.show(block=False)


class MLP(torch.nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = torch.nn.Linear(1, hidden)
        self.fc2 = torch.nn.Linear(hidden, 1)
        self.activation = PiecewiseActivation()

    def forward(self, x):
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        return x

# activation = PiecewiseActivation()
# model = MLP([1, hidden, 1], activation=activation)
model = MLP()
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
