# backend/app/services/sr_model_definition.py
import torch
import torch.nn as nn
from torchvision import models # Keep this import for resnet50 structure
import torchvision.transforms.functional as TF

class PixelShuffleBlock(nn.Module):
    def __init__(self, in_channels, out_channels, upscale_factor):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels * (upscale_factor ** 2), kernel_size=3, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)
        self.relu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        return self.relu(self.pixel_shuffle(self.conv(x)))

class ConvBlock(nn.Module):
    """Decoder block with NO BatchNorm to prevent statistical corruption."""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, 1)
        self.relu1 = nn.LeakyReLU(0.2, inplace=True)
        self.dropout = nn.Dropout(0.2)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1)
        self.relu2 = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        x = self.relu1(self.conv1(x))
        x = self.dropout(x)
        return self.relu2(self.conv2(x))

class SRModel4x(nn.Module):
    def __init__(self, upscale_factor=4, encoder_pretrained=False):
        super().__init__()
        self.upscale_factor = upscale_factor
        resnet = models.resnet50(weights=None)
        self.register_buffer('imagenet_mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('imagenet_std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        self.encoder_conv1, self.encoder_bn1, self.encoder_relu = resnet.conv1, resnet.bn1, resnet.relu
        self.encoder_maxpool = resnet.maxpool
        self.encoder_layer1, self.encoder_layer2, self.encoder_layer3, self.encoder_layer4 = resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4
        self.bottleneck_conv = ConvBlock(2048, 1024)
        self.dec_up4 = PixelShuffleBlock(1024, 512, 2)
        self.dec_conv4 = ConvBlock(512 + 1024, 512)
        self.dec_up3 = PixelShuffleBlock(512, 256, 2)
        self.dec_conv3 = ConvBlock(256 + 512, 256)
        self.dec_up2 = PixelShuffleBlock(256, 128, 2)
        self.dec_conv2 = ConvBlock(128 + 256, 128)
        self.dec_up1 = PixelShuffleBlock(128, 64, 2)
        self.dec_conv1 = ConvBlock(64 + 64, 64)
        self.final_up = PixelShuffleBlock(64, 32, 2)
        modules, current_channels, temp_factor = [], 32, self.upscale_factor
        while temp_factor > 1 and temp_factor % 2 == 0:
            out_ch = current_channels // 2
            modules.append(PixelShuffleBlock(current_channels, out_ch, 2))
            current_channels = out_ch
            temp_factor //= 2
        self.final_scaling_ops = nn.Sequential(*modules)
        self.final_conv_output = nn.Conv2d(current_channels, 3, 3, 1, 1)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x_0_to_1 = (x * 0.5) + 0.5
        x_norm = (x_0_to_1 - self.imagenet_mean) / self.imagenet_std
        c1 = self.encoder_relu(self.encoder_bn1(self.encoder_conv1(x_norm)))
        p1 = self.encoder_maxpool(c1)
        e1 = self.encoder_layer1(p1)
        e2 = self.encoder_layer2(e1)
        e3 = self.encoder_layer3(e2)
        e4 = self.encoder_layer4(e3)
        d_bn = self.bottleneck_conv(e4)
        d4_up = self.dec_up4(d_bn)
        if d4_up.shape[2:] != e3.shape[2:]:
            e3 = TF.resize(e3, size=d4_up.shape[2:])
        d4 = self.dec_conv4(torch.cat([d4_up, e3], 1))
        d3_up = self.dec_up3(d4)
        if d3_up.shape[2:] != e2.shape[2:]:
            e2 = TF.resize(e2, size=d3_up.shape[2:])
        d3 = self.dec_conv3(torch.cat([d3_up, e2], 1))
        d2_up = self.dec_up2(d3)
        if d2_up.shape[2:] != e1.shape[2:]:
            e1 = TF.resize(e1, size=d2_up.shape[2:])
        d2 = self.dec_conv2(torch.cat([d2_up, e1], 1))
        d1_up = self.dec_up1(d2)
        if d1_up.shape[2:] != c1.shape[2:]:
            c1 = TF.resize(c1, size=d1_up.shape[2:])
        d1 = self.dec_conv1(torch.cat([d1_up, c1], 1))
        d0 = self.final_up(d1)
        if d0.shape[2:] != x.shape[2:]:
            d0 = TF.resize(d0, size=x.shape[2:])
        scaled = self.final_scaling_ops(d0)
        out = self.final_conv_output(scaled)
        return self.tanh(out)

# --- 2x Model (original) ---
class ConvBlock2x(nn.Module):
    def __init__(self, in_channels, out_channels, use_bn=True):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels) if use_bn else nn.Identity()
        self.relu1 = nn.LeakyReLU(0.2, inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels) if use_bn else nn.Identity()
        self.relu2 = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        return x

class SRModel(nn.Module):
    def __init__(self, upscale_factor=2, encoder_pretrained=False):
        super().__init__()
        self.upscale_factor = upscale_factor
        resnet = models.resnet50(weights=None)
        self.encoder_conv1 = resnet.conv1
        self.encoder_bn1 = resnet.bn1
        self.encoder_relu = resnet.relu
        self.encoder_maxpool = resnet.maxpool
        self.encoder_layer1 = resnet.layer1
        self.encoder_layer2 = resnet.layer2
        self.encoder_layer3 = resnet.layer3
        self.encoder_layer4 = resnet.layer4
        self.bottleneck_conv = ConvBlock2x(2048, 1024)
        self.dec_up4 = PixelShuffleBlock(1024, 512, upscale_factor=2)
        self.dec_conv4 = ConvBlock2x(512 + 1024, 512)
        self.dec_up3 = PixelShuffleBlock(512, 256, upscale_factor=2)
        self.dec_conv3 = ConvBlock2x(256 + 512, 256)
        self.dec_up2 = PixelShuffleBlock(256, 128, upscale_factor=2)
        self.dec_conv2 = ConvBlock2x(128 + 256, 128)
        self.dec_up1 = PixelShuffleBlock(128, 64, upscale_factor=2)
        self.dec_conv1 = ConvBlock2x(64 + 64, 64)
        self.final_up = PixelShuffleBlock(64, 32, upscale_factor=2)
        modules_for_final_scaling = []
        current_channels = 32
        if self.upscale_factor == 2:
            modules_for_final_scaling.append(PixelShuffleBlock(current_channels, max(16, current_channels // 2), upscale_factor=2))
            current_channels = max(16, current_channels // 2)
        elif self.upscale_factor == 4:
            modules_for_final_scaling.append(PixelShuffleBlock(current_channels, max(16, current_channels // 2), upscale_factor=2))
            current_channels = max(16, current_channels // 2)
            modules_for_final_scaling.append(PixelShuffleBlock(current_channels, max(16, current_channels // 2), upscale_factor=2))
            current_channels = max(16, current_channels // 2)
        self.final_scaling_ops = nn.Sequential(*modules_for_final_scaling)
        self.final_conv_output = nn.Conv2d(current_channels, 3, kernel_size=3, padding=1)
        self.tanh = nn.Tanh()

    def forward(self, x):
        c1 = self.encoder_relu(self.encoder_bn1(self.encoder_conv1(x)))
        p1 = self.encoder_maxpool(c1)
        e1 = self.encoder_layer1(p1)
        e2 = self.encoder_layer2(e1)
        e3 = self.encoder_layer3(e2)
        e4 = self.encoder_layer4(e3)
        d_bn = self.bottleneck_conv(e4)
        up4 = self.dec_up4(d_bn)
        if e3.shape[2:] != up4.shape[2:]:
            e3 = TF.resize(e3, up4.shape[2:], interpolation=TF.InterpolationMode.BILINEAR)
        d4 = self.dec_conv4(torch.cat([up4, e3], 1))
        up3 = self.dec_up3(d4)
        if e2.shape[2:] != up3.shape[2:]:
            e2 = TF.resize(e2, up3.shape[2:], interpolation=TF.InterpolationMode.BILINEAR)
        d3 = self.dec_conv3(torch.cat([up3, e2], 1))
        up2 = self.dec_up2(d3)
        if e1.shape[2:] != up2.shape[2:]:
            e1 = TF.resize(e1, up2.shape[2:], interpolation=TF.InterpolationMode.BILINEAR)
        d2 = self.dec_conv2(torch.cat([up2, e1], 1))
        up1 = self.dec_up1(d2)
        if c1.shape[2:] != up1.shape[2:]:
            c1 = TF.resize(c1, up1.shape[2:], interpolation=TF.InterpolationMode.BILINEAR)
        d1 = self.dec_conv1(torch.cat([up1, c1], 1))
        final_up = self.final_up(d1)
        scaled = self.final_scaling_ops(final_up)
        out = self.final_conv_output(scaled)
        return self.tanh(out)

# --- 2x Model (Crazy) - Model 2 Architecture ---
class PixelShuffleBlock2x(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels * 4, kernel_size=3, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor=2)
        self.relu = nn.LeakyReLU(0.2, inplace=True)
    def forward(self, x):
        return self.relu(self.pixel_shuffle(self.conv(x)))

class ConvBlockBN(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.LeakyReLU(0.2, inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.LeakyReLU(0.2, inplace=True)
    def forward(self, x):
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        return x

class SRModel2xCrazy(nn.Module):
    def __init__(self):
        super().__init__()
        resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        for param in resnet.parameters():
            param.requires_grad = False
        self.encoder_conv1 = resnet.conv1
        self.encoder_bn1 = resnet.bn1
        self.encoder_relu = resnet.relu
        self.encoder_maxpool = resnet.maxpool
        self.encoder_layer1 = resnet.layer1
        self.encoder_layer2 = resnet.layer2
        self.encoder_layer3 = resnet.layer3
        self.encoder_layer4 = resnet.layer4
        self.bottleneck_conv = ConvBlockBN(2048, 1024)
        self.dec_up4 = PixelShuffleBlock2x(1024, 512)
        self.dec_conv4 = ConvBlockBN(512 + 1024, 512)
        self.dec_up3 = PixelShuffleBlock2x(512, 256)
        self.dec_conv3 = ConvBlockBN(256 + 512, 256)
        self.dec_up2 = PixelShuffleBlock2x(256, 128)
        self.dec_conv2 = ConvBlockBN(128 + 256, 128)
        self.dec_up1 = PixelShuffleBlock2x(128, 64)
        self.dec_conv1 = ConvBlockBN(64 + 64, 64)
        self.final_up = PixelShuffleBlock2x(64, 32)
        self.superres_up = PixelShuffleBlock2x(32, 16)
        self.final_conv_output = nn.Conv2d(16, 3, kernel_size=3, padding=1)
        self.tanh = nn.Tanh()

    def forward(self, x):
        # Encoder Path
        c1 = self.encoder_relu(self.encoder_bn1(self.encoder_conv1(x)))
        e1 = self.encoder_layer1(self.encoder_maxpool(c1))
        e2 = self.encoder_layer2(e1)
        e3 = self.encoder_layer3(e2)
        e4 = self.encoder_layer4(e3)

        d_bn = self.bottleneck_conv(e4)

        # Decoder Path with Self-Correcting Crop
        d4_up = self.dec_up4(d_bn)
        e3_cropped = TF.center_crop(e3, d4_up.shape[2:])
        d4 = self.dec_conv4(torch.cat([d4_up, e3_cropped], 1))

        d3_up = self.dec_up3(d4)
        e2_cropped = TF.center_crop(e2, d3_up.shape[2:])
        d3 = self.dec_conv3(torch.cat([d3_up, e2_cropped], 1))

        d2_up = self.dec_up2(d3)
        e1_cropped = TF.center_crop(e1, d2_up.shape[2:])
        d2 = self.dec_conv2(torch.cat([d2_up, e1_cropped], 1))

        d1_up = self.dec_up1(d2)
        c1_cropped = TF.center_crop(c1, d1_up.shape[2:])
        d1 = self.dec_conv1(torch.cat([d1_up, c1_cropped], 1))
        
        # Final Upscaling
        d0_up = self.final_up(d1)
        sr_up = self.superres_up(d0_up)
        out = self.final_conv_output(sr_up)
        final_out = self.tanh(out)
        
        return final_out

# --- 8x Model (Model 5) - 8x Architecture ---
class SRModel8x(nn.Module):
    def __init__(self, upscale_factor=8, encoder_pretrained=False):
        super().__init__()
        self.upscale_factor = upscale_factor
        resnet = models.resnet50(weights=None)
        self.register_buffer('imagenet_mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('imagenet_std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        self.encoder_conv1, self.encoder_bn1, self.encoder_relu = resnet.conv1, resnet.bn1, resnet.relu
        self.encoder_maxpool = resnet.maxpool
        self.encoder_layer1, self.encoder_layer2, self.encoder_layer3, self.encoder_layer4 = resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4
        self.bottleneck_conv = ConvBlock(2048, 1024)
        self.dec_up4 = PixelShuffleBlock(1024, 512, 2)
        self.dec_conv4 = ConvBlock(512 + 1024, 512)
        self.dec_up3 = PixelShuffleBlock(512, 256, 2)
        self.dec_conv3 = ConvBlock(256 + 512, 256)
        self.dec_up2 = PixelShuffleBlock(256, 128, 2)
        self.dec_conv2 = ConvBlock(128 + 256, 128)
        self.dec_up1 = PixelShuffleBlock(128, 64, 2)
        self.dec_conv1 = ConvBlock(64 + 64, 64)
        self.dec_up_pre = PixelShuffleBlock(64, 64, 2)
        self.dec_conv_pre = ConvBlock(64, 32)
        modules, current_channels, temp_factor = [], 32, self.upscale_factor
        while temp_factor > 1 and temp_factor % 2 == 0:
            out_ch = max(16, current_channels // 2)
            modules.append(PixelShuffleBlock(current_channels, out_ch, 2))
            current_channels = out_ch
            temp_factor //= 2
        self.final_scaling_ops = nn.Sequential(*modules)
        self.final_conv_output = nn.Conv2d(current_channels, 3, 3, 1, 1)

    def forward(self, x):
        x_0_to_1 = (x * 0.5) + 0.5
        x_norm = (x_0_to_1 - self.imagenet_mean) / self.imagenet_std
        c1 = self.encoder_relu(self.encoder_bn1(self.encoder_conv1(x_norm)))
        p1 = self.encoder_maxpool(c1)
        e1 = self.encoder_layer1(p1)
        e2 = self.encoder_layer2(e1)
        e3 = self.encoder_layer3(e2)
        e4 = self.encoder_layer4(e3)
        d_bn = self.bottleneck_conv(e4)
        d4_up = self.dec_up4(d_bn)
        if d4_up.shape[2:] != e3.shape[2:]:
            e3 = TF.resize(e3, size=d4_up.shape[2:])
        d4 = self.dec_conv4(torch.cat([d4_up, e3], 1))
        d3_up = self.dec_up3(d4)
        if d3_up.shape[2:] != e2.shape[2:]:
            e2 = TF.resize(e2, size=d3_up.shape[2:])
        d3 = self.dec_conv3(torch.cat([d3_up, e2], 1))
        d2_up = self.dec_up2(d3)
        if d2_up.shape[2:] != e1.shape[2:]:
            e1 = TF.resize(e1, size=d2_up.shape[2:])
        d2 = self.dec_conv2(torch.cat([d2_up, e1], 1))
        d1_up = self.dec_up1(d2)
        if d1_up.shape[2:] != c1.shape[2:]:
            c1 = TF.resize(c1, size=d1_up.shape[2:])
        d1 = self.dec_conv1(torch.cat([d1_up, c1], 1))
        d0 = self.dec_up_pre(d1)
        if d0.shape[2:] != x.shape[2:]:
            d0 = TF.resize(d0, size=x.shape[2:])
        d0_refined = self.dec_conv_pre(d0)
        scaled = self.final_scaling_ops(d0_refined)
        out = self.final_conv_output(scaled)
        return out