import torch
from torch import nn
from torch.nn import init

class ECAChannelAttention(nn.Module):
    def __init__(self, channel, gamma=2, b=1):
        """
        Enhanced Channel Attention Mechanism (ECA).
        Args:
            channel (int): Number of input channels.
            gamma (int): Gamma value to compute adaptive kernel size. Default is 2.
            b (int): Bias for kernel size computation. Default is 1.
        """
        super().__init__()
        # Adaptive kernel size calculation
        kernel_size = int(abs((torch.log2(torch.tensor(channel, dtype=torch.float32)) + b) // gamma))
        kernel_size = kernel_size + 1 if kernel_size % 2 == 0 else kernel_size
        self.conv1d = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Global Average Pooling (GAP)
        avg_pool = torch.mean(x, dim=(2, 3), keepdim=True)  # Shape: (B, C, 1, 1)
        # Squeeze and Excite with 1D Conv
        avg_pool = avg_pool.squeeze(-1).transpose(-1, -2)  # Shape: (B, 1, C)
        attention = self.conv1d(avg_pool).transpose(-1, -2).unsqueeze(-1)  # Shape: (B, C, 1, 1)
        # Sigmoid activation
        return self.sigmoid(attention) * x

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_result, _ = torch.max(x, dim=1, keepdim=True)
        avg_result = torch.mean(x, dim=1, keepdim=True)
        result = torch.cat([max_result, avg_result], 1)
        output = self.conv(result)
        output = self.sigmoid(output)
        return output


class CBAMBlock(nn.Module):
    def __init__(self, channel=512, reduction=16, kernel_size=49):
        super().__init__()
        self.ca = ECAChannelAttention(channel=channel)  # Use ECA instead of original CAM
        self.sa = SpatialAttention(kernel_size=kernel_size)

        # import pdb;pdb.set_trace()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        b, c, _, _ = x.size()
        residual = x
        out = x * self.ca(x)
        out = out * self.sa(out)
        return out + residual


if __name__ == '__main__':
    input = torch.randn(1, 768, 8, 8)
    kernel_size = input.shape[2]
    cbam = CBAMBlock(channel=768, reduction=16, kernel_size=7)
    output = cbam(input)
    print(output.shape)

