import torch
import torch.nn as nn
import torch.nn.functional as F

class _LRASPP(nn.Module):
    """Lite R-ASPP"""

    def __init__(self, in_channels, norm_layer, **kwargs):
        super(_LRASPP, self).__init__()
        out_channels = 128
        self.b0 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            norm_layer(out_channels),
            nn.ReLU(True)
        )

        self.b1 = nn.Sequential(
            #nn.AvgPool2d(kernel_size=(49, 49), stride=(16, 20),count_include_pad=True),  # check it
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        size = x.size()[2:]
        feat1 = self.b0(x)
        feat2 = F.avg_pool2d(input=x,kernel_size=(49, 49), stride=(16, 20),padding=0, ceil_mode=False, count_include_pad=True,divisor_override=True)
        feat2 = self.b1(x)
        feat2 = F.interpolate(feat2, size, mode='bilinear', align_corners=True)
        x = feat1 * feat2  # check it
        return x


if __name__ == '__main__':
    net = _LRASPP(in_channels=576,norm_layer=nn.BatchNorm2d)

    input_size=(1, 576,64, 32)
    net=net.cuda()
    
    x = torch.randn(input_size)
    x=x.cuda()
    
    net.eval()
    for i in range(10):
        out = net(x)
    print(out.shape)
