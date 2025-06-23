from torch import nn
import torch


# 计算 Kullback-Leibler 散度
def kl_divergence(p, q):
    p = p + 1e-10  # 避免 log(0) 错误
    q = q + 1e-10
    return torch.sum(p * torch.log(p / q), dim=1).mean()  # 归一化


# 计算 Jensen-Shannon 散度
def js_divergence(p, q):
    m = 0.5 * (p + q)
    return 0.5 * (kl_divergence(p, m) + kl_divergence(q, m))


# 自定义损失函数
class CustomLoss(nn.Module):
    def __init__(self, mae_fac=1.0, mse_fac=1.0, kl_fac=1.0, js_fac=1.0):  # original:mae_fac=1.0, mse_fac=1.0, kl_fac=1.0, js_fac=1.0
        super(CustomLoss, self).__init__()
        self.mae_fac = mae_fac
        self.mse_fac = mse_fac
        self.kl_fac = kl_fac
        self.js_fac = js_fac
        self.mae_loss = nn.L1Loss()
        self.mse_loss = nn.MSELoss()

    def forward(self, output, target):
        mae = self.mae_loss(output, target)
        mse = self.mse_loss(output, target)

        # 确保张量值在 [0, 1] 范围内
        # output = torch.clamp(output, 0, 1)
        # target = torch.clamp(target, 0, 1)

        # 计算 KL 和 JS 散度
        kl = kl_divergence(output, target)
        js = js_divergence(output, target)

        # 确保损失值为正数
        loss = self.mae_fac * mae + self.mse_fac * mse + self.kl_fac * torch.relu(kl) + self.js_fac * torch.relu(js)
        return loss


class ConvAutoencoder(nn.Module):
    def __init__(self, input_channels=3):  # input_channels与处理的数据集图像的通道数保持一致即可
        super(ConvAutoencoder, self).__init__()

        # encoder
        self.enconv1 = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1)
        self.enconv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.enconv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.enconv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.enconv5 = nn.Conv2d(256, 512, kernel_size=2, stride=2)
        self.relu = nn.ReLU()
        self.max_pool2d = nn.MaxPool2d((2, 2), stride=2)
        self.encoder = nn.Sequential(
            # nn.Conv2d(input_channels, 16, kernel_size=3, stride=2, padding=1),  # [B, 3, 32, 32] -> [B, 16, 16, 16]
            # nn.ReLU(),
            # nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),  # [B, 16, 16, 16] -> [B, 32, 8, 8]
            # nn.ReLU(),
            # nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # [B, 32, 8, 8] -> [B, 64, 4, 4]
            # nn.ReLU(),
            # nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # [B, 64, 4, 4] -> [B, 128, 2, 2]
            # nn.ReLU(),
            # nn.Conv2d(128, 256, kernel_size=2, stride=2)  # [B, 128, 2, 2] -> [B, 256, 1, 1]
            self.enconv1,
            self.relu,
            self.max_pool2d,  # [B, 3, 32, 32] -> [B, 32, 16, 16]
            self.enconv2,
            self.relu,
            self.max_pool2d,  # [B, 32, 16, 16] -> [B, 64, 8, 8]
            self.enconv3,
            self.relu,
            self.max_pool2d,  # [B, 64, 8, 8] -> [B, 128, 4, 4]
            self.enconv4,
            self.relu,
            self.max_pool2d,  # [B, 128, 4, 4] -> [B, 256, 2, 2]
            self.enconv5,  # [B, 256, 2, 2] -> [B, 512, 1, 1]
            self.relu
        )

        # decoder
        self.deconv1 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)  # [B, 512, 1, 1] -> [B, 256, 2, 2]
        self.deconv2 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1,
                                          output_padding=1)  # [B, 256, 2, 2] -> [B, 128, 4, 4]
        self.deconv3 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1,
                                          output_padding=1)  # [B, 128, 4, 4] -> [B, 64, 8, 8]
        self.deconv4 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1,
                                          output_padding=1)  # [B, 64, 8, 8] -> [B, 32, 16, 16]
        self.deconv5 = nn.ConvTranspose2d(32, input_channels, kernel_size=3, stride=2, padding=1,
                                          output_padding=1)  # [B, 32, 16, 16] -> [B, 3, 32, 32]

        self.sigmoid = nn.Sigmoid()

        self.decoder = nn.Sequential(
            self.deconv1,
            self.relu,
            self.deconv2,
            self.relu,
            self.deconv3,
            self.relu,
            self.deconv4,
            self.relu,
            self.deconv5,
            self.sigmoid
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    def feature_list(self, x):
        encoder_features = []
        decoder_features = []

        # x = self.encoder(x)
        # Encode with intermediate outputs
        en_out0 = self.max_pool2d(self.relu(self.enconv1(x)))
        encoder_features.append(en_out0)

        en_out1 = self.max_pool2d(self.relu(self.enconv2(en_out0)))
        encoder_features.append(en_out1)

        en_out2 = self.max_pool2d(self.relu(self.enconv3(en_out1)))
        encoder_features.append(en_out2)

        en_out3 = self.max_pool2d(self.relu(self.enconv4(en_out2)))
        encoder_features.append(en_out3)

        en_out4 = self.relu(self.enconv5(en_out3))
        encoder_features.append(en_out4)

        # Decoder with intermediate outputs
        de_out0 = en_out4
        decoder_features.append(de_out0)
        de_out1 = self.relu(self.deconv1(de_out0))
        decoder_features.append(de_out1)

        de_out2 = self.relu(self.deconv2(de_out1))
        decoder_features.append(de_out2)

        de_out3 = self.relu(self.deconv3(de_out2))
        decoder_features.append(de_out3)

        de_out4 = self.relu(self.deconv4(de_out3))
        decoder_features.append(de_out4)

        de_out5 = self.sigmoid(self.deconv5(de_out4))
        # decoder_features.append(de_out5)

        return encoder_features, decoder_features
