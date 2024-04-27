import torch


class PiecewiseActivation(torch.nn.Module):
    def __init__(self, n_points=10, left=-1.0, right=1.0):
        super().__init__()
        self.xs = torch.linspace(left, right, n_points)
        # self.register_buffer("xs", self.xs)
        self.slopes = torch.nn.Parameter(torch.tensor([0.0, 1.0]))
        scale = (right - left) / (n_points - 1)
        self.ys = torch.nn.Parameter(torch.cumsum(torch.randn(n_points) * scale, dim=0))
        print(self.ys)

    def forward(self, x):
        in_shape = x.shape
        x = x.flatten()

        diff = x.unsqueeze(1) - self.xs.unsqueeze(0)  # (batch, n_points)
        left_endpt = torch.argmin(torch.where(diff > 0, diff, torch.inf), dim=1)
        right_endpt = torch.argmax(torch.where(diff < 0, diff, -torch.inf), dim=1)

        out = self.ys[left_endpt] + (x - self.xs[left_endpt]) * (self.ys[right_endpt] - self.ys[left_endpt]) / (
            self.xs[right_endpt] - self.xs[left_endpt]
        )
        out = torch.where(x < self.xs[0], self.ys[0] - (self.xs[0] - x) * self.slopes[0], out)
        out = torch.where(x > self.xs[-1], self.ys[-1] + (x - self.xs[-1]) * self.slopes[1], out)

        out = out.reshape(in_shape)
        return out
