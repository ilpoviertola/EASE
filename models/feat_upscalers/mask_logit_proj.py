from torch import nn
from torch import Tensor
from torch.nn import functional as F


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Conv2d(n, k, kernel_size=1, stride=1, padding=0)
            for n, k in zip([input_dim] + h, h + [output_dim])
        )

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class MaskLogitMLP(nn.Module):
    def __init__(self, num_q: int):
        super().__init__()
        self.mlp = MLP(num_q, 4 * num_q, num_q, 3)

    def forward(self, mask_logits: Tensor, mask_feature: Tensor) -> Tensor:
        mask_logits = self.mlp(mask_logits)
        return mask_logits


class MaskLogitProj(nn.Module):
    def __init__(self, num_q: int, dim: int, scale_factor: int = 4):
        super().__init__()
        self.scale_factor = scale_factor

        self.mlp = MLP(num_q, 4 * dim, dim, 3)
        self.conv_1 = nn.Conv2d(dim, dim // 2, kernel_size=3, stride=1, padding=1)
        self.conv_2 = nn.Conv2d(dim // 2, dim // 4, kernel_size=3, stride=1, padding=1)
        self.act = nn.ReLU(True)
        self.conv_3 = nn.Conv2d(
            dim // 4, num_q, kernel_size=1, stride=1, padding=0, bias=False
        )

    def forward(self, mask_logits: Tensor, mask_feature: Tensor) -> Tensor:
        mask_logits = self.mlp(mask_logits)
        x = mask_feature + mask_logits
        x = self.conv_1(x)
        x = F.interpolate(
            x, scale_factor=self.scale_factor, mode="bilinear", align_corners=False
        )
        x = self.conv_2(x)
        x = self.act(x)
        x = self.conv_3(x)
        return x
