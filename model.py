import torch
import torch.nn as nn
from torch.nn import init

def Init_Weights(net, init_type, gain):
    print('Init Network Weights')

    def init_func(m):
        classname = m.__class__.__name__

        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=gain)

        if hasattr(m, 'bias') and m.bias is not None:
            init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1 or classname.find('BatchNorm1d') != -1:
            init.normal_(m.weight.data, 1.0, 0.02)
            init.constant_(m.bias.data, 0.0)

    print(f'Initialize network with {init_type}')
    net.apply(init_func)

class SEAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(SEAttention, self).__init__()
        self.fc1 = nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1)
        self.fc2 = nn.Conv2d(in_channels // reduction, in_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        se_weight = torch.mean(x, dim=(2, 3), keepdim=True)  # Squeeze
        se_weight = self.fc1(se_weight)
        se_weight = nn.ReLU()(se_weight)
        se_weight = self.fc2(se_weight)
        se_weight = self.sigmoid(se_weight)  # Excitation
        return x * se_weight  # Apply attention

class MUNet(nn.Module):
    def __init__(self, band, num_classes, ldr_dim, reduction):
        super(MUNet, self).__init__()
        self.fc_hsi = nn.Sequential(
            nn.Conv2d(band, max(band//2, 1), kernel_size=1),  # Ensure at least 1 channel
            nn.BatchNorm2d(max(band//2, 1)),
            nn.ReLU(),
            nn.Conv2d(max(band//2, 1), max(band//4, 1), kernel_size=1),
            nn.BatchNorm2d(max(band//4, 1)),
            nn.ReLU(),
            nn.Conv2d(max(band//4, 1), num_classes, kernel_size=1)
        )

        self.fc_lidar = nn.Sequential(
            nn.Conv2d(ldr_dim, max(ldr_dim // 2, 1), kernel_size=3, padding=1),  # Ensure at least 1 channel
            nn.BatchNorm2d(max(ldr_dim // 2, 1)),
            nn.ReLU(),
            nn.Conv2d(max(ldr_dim // 2, 1), max(ldr_dim // 4, 1), kernel_size=3, padding=1),
            nn.BatchNorm2d(max(ldr_dim // 4, 1)),
            nn.ReLU(),
            nn.Conv2d(max(ldr_dim // 4, 1), num_classes, kernel_size=1)        
       )

        self.se_attention = SEAttention(num_classes)

        self.fc_fused = nn.Sequential(
            nn.Conv2d(num_classes, num_classes // 2, kernel_size=1),
            nn.BatchNorm2d(num_classes // 2),
            nn.ReLU(),
            nn.Conv2d(num_classes // 2, num_classes, kernel_size=1)
        )

        self.decoder = nn.Sequential(
            nn.Conv2d(num_classes, band, kernel_size=1, bias=False),
            nn.ReLU()
        )

    def forward(self, hsi, lidar):
        hsi_features = self.fc_hsi(hsi)
        lidar_features = self.fc_lidar(lidar)

        # Apply SE attention on LiDAR features
        lidar_features = self.se_attention(lidar_features)

        # Element-wise multiplication (fusion) of HSI and LiDAR features
        fused_features = hsi_features * lidar_features  # Element-wise product

        # Further process fused features
        fused_output = self.fc_fused(fused_features)

        # Abundance output
        abundances = fused_output

        # Reconstruct HSI (Endmembers prediction)
        hsi_reconstructed = self.decoder(abundances)

        return abundances, hsi_reconstructed
