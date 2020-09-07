import torch.nn as nn


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.decoder = nn.Sequential(
                        nn.Conv2d(1024, 128, 3, 1, 1, bias=False),
                        nn.BatchNorm2d(128),
                        nn.ReLU(),
                        nn.Upsample(14),
                        nn.Conv2d(128, 64, 3, 1, 1, bias=False),
                        nn.BatchNorm2d(64),
                        nn.ReLU(),
                        nn.Upsample(28),
                        nn.Conv2d(64, 32, 3, 1, 1, bias=False),
                        nn.BatchNorm2d(32),
                        nn.ReLU(),
                        nn.Upsample(56),
                        nn.Conv2d(32, 16, 3, 1, 1, bias=False),
                        nn.BatchNorm2d(16),
                        nn.ReLU(),
                        nn.Upsample(112),
                        nn.Conv2d(16, 16, 3, 1, 1, bias=False),
                        nn.BatchNorm2d(16),
                        nn.ReLU(),
                        nn.Upsample(224),
                        nn.Conv2d(16, 3, 3, 1, 1, bias=False),
                        nn.Sigmoid(),
        )

    def forward(self, x):
        return self.decoder(x)
