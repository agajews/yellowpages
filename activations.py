import torch


class PiecewiseActivation(torch.nn.Module):
    def __init__(self, n_points=10):
        super().__init__()
        self.xs = torch.linspace(-5, 5, n_points)
        # self.register_buffer("xs", self.xs)
        self.slopes = torch.nn.Parameter(torch.tensor([0.0, 1.0]))
        self.ys = torch.nn.Parameter(torch.cumsum(torch.randn(n_points), dim=0))

    def forward(self, x):
        in_shape = x.shape
        x = x.flatten()

        left_endpt = torch.argmin(x.unsqueeze(1) - self.xs.unsqueeze(0), dim=1)
        right_endpt = torch.argmin(self.xs.unsqueeze(0) - x.unsqueeze(1), dim=1)

        out = self.ys[left_endpt] + (x - self.xs[left_endpt]) * (self.ys[right_endpt] - self.ys[left_endpt]) / (
            self.xs[right_endpt] - self.xs[left_endpt]
        )
        out = torch.where(x < self.xs[0], self.ys[0] - (self.xs[0] - x) * self.slopes[0], out)
        out = torch.where(x > self.xs[-1], self.ys[-1] + (x - self.xs[-1]) * self.slopes[1], out)

        out = out.reshape(in_shape)
        return out
