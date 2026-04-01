import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


class UNet(nn.Module):
    def __init__(self, in_channels=22, out_channels=2, features=[32, 64, 128, 256]):
        super().__init__()
        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.pool = nn.MaxPool2d(2)

        # Encoder
        ch = in_channels
        for f in features:
            self.encoders.append(ConvBlock(ch, f))
            ch = f

        # Bottleneck
        self.bottleneck = ConvBlock(features[-1], features[-1] * 2)

        # Decoder
        for f in reversed(features):
            self.decoders.append(nn.ConvTranspose2d(f * 2, f, 2, stride=2))
            self.decoders.append(ConvBlock(f * 2, f))

        # Final output
        self.final = nn.Conv2d(features[0], out_channels, 1)
        self.sigmoid = nn.Sigmoid()  # outputs in [0,1]

    def forward(self, x):
        skip_connections = []

        # Encoder path
        for encoder in self.encoders:
            x = encoder(x)
            skip_connections.append(x)
            x = self.pool(x)

        # Bottleneck
        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        # Decoder path
        for i in range(0, len(self.decoders), 2):
            x = self.decoders[i](x)       # upsample
            skip = skip_connections[i//2]

            # Handle size mismatch
            if x.shape != skip.shape:
                x = torch.nn.functional.interpolate(
                    x, size=skip.shape[2:]
                )

            x = torch.cat([skip, x], dim=1)
            x = self.decoders[i+1](x)     # conv block

        return self.sigmoid(self.final(x))