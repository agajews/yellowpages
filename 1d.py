import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import time

from components import MLP


hidden = 100
lr = 0.0001


xs = torch.rand(size=(100, 1)) * 10
val_xs = torch.rand(size=(100, 1)) * 10
print(xs)
ys = torch.sin(xs)

plt.ion()
plt.show(block=False)


# class MLP(torch.nn.Module):
#     def __init__(self):
#         super(MLP, self).__init__()
#         self.fc1 = torch.nn.Linear(1, hidden)
#         self.fc2 = torch.nn.Linear(hidden, 1)

#     def forward(self, x):
#         x = self.fc1(x)
#         x = F.relu(x)
#         x = self.fc2(x)
#         return x

model = MLP([1, hidden, hidden, 1])
optimizer = torch.optim.Adam(model.parameters(), lr=lr)


def visualize():
    with torch.no_grad():
        val_y_preds = model(val_xs)

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
    optimizer.step()

    if epoch % 10 == 0:
        visualize()