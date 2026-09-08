from torch import nn


class LateralConvScaler(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(LateralConvScaler, self).__init__()
        self.in_proj = nn.Conv2d(
            in_channels, out_channels, kernel_size=1, stride=1, padding=0
        )
        self.in_norm = nn.GroupNorm(32, out_channels)
        self.out_proj = nn.Conv2d(
            out_channels, out_channels, kernel_size=1, stride=1, padding=0
        )
        self.out_norm = nn.GroupNorm(32, out_channels)

    def forward(self, x):
        x = self.in_proj(x)
        x = self.in_norm(x)
        x = self.out_proj(x)
        x = self.out_norm(x)
        return x
