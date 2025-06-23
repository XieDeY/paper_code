import torch
import torch.nn as nn
import torch.nn.functional as F


class Text_CNN_ResNet(torch.nn.Module):
    # Text-CNN Detector For ResNet-34
    def __init__(self):  # 卷积层调整通道数， 最大池化层调整高宽
        super(Text_CNN_ResNet, self).__init__()
        self.cp0 = nn.Sequential(  # [B, 32, 16, 16] -> [B, 64, 8, 8]
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.MaxPool2d((2, 2), stride=2)
        )
        self.cp1 = nn.Sequential(  # [B, 64, 8, 8] -> [B, 128, 4, 4]
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.MaxPool2d((2, 2), stride=2)
        )
        self.cp2 = nn.Sequential(  # [B, 128, 4, 4] -> [B, 256, 2, 2]
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.MaxPool2d((2, 2), stride=2)
        )
        self.cp3 = nn.Sequential(  # [B, 256, 2, 2] -> [B, 512, 1, 1]
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.MaxPool2d((2, 2), stride=2)
        )

        filter_sizes = [1, 2, 3, 4, 5]  # 定义文本卷积核大小
        num_filters = 100  # 定义每个文本卷积核的输出通道数（个数）
        self.convs = nn.ModuleList([nn.Conv2d(1, num_filters, (K, 512), bias=True) for K in filter_sizes])
        self.dropout1 = nn.Dropout(0.1)
        self.fc1 = nn.Linear(len(filter_sizes) * num_filters, 200)
        self.dropout2 = nn.Dropout(0.5)  # 防止过拟合：1.表示每个神经元有0.5的概率可能性不被激活；2.将tensor中的部分值置为0，模拟现实中的部分数据缺失。
        self.fc2 = nn.Linear(200, 2)

    def forward(self, x):
        out0 = self.cp0(x[0])
        out0 = self.cp1(out0)
        out0 = self.cp2(out0)
        out0 = self.cp3(out0)
        out0 = out0.view(out0.size(0), 1, -1)  # 将out0变换为一个三维张量，第一位保持batch_size数，第二个维度为1（展平），第三个维度根据前两个维度自动推算。

        out1 = self.cp1(x[1])
        out1 = self.cp2(out1)
        out1 = self.cp3(out1)
        out1 = out1.view(out1.size(0), 1, -1)

        out2 = self.cp2(x[2])
        out2 = self.cp3(out2)
        out2 = out2.view(out2.size(0), 1, -1)

        out3 = self.cp3(x[3])
        out3 = out3.view(out3.size(0), 1, -1)

        out4 = x[4].view(x[4].size(0), 1, -1)

        txt = torch.cat((out0, out1, out2, out3, out4), 1)  # 从第1维进行cat，将同一个样本的在不同神经网络层的特征图进行cat。
        # txt = torch.cat((out0, ), 1)  # 从第1维进行cat，将同一个样本的在不同神经网络层的特征图进行cat。
        txt = torch.unsqueeze(txt, 1)  # 增加通道维度(3D ——> 4D)

        # print(out0.size())
        # print(out1.size())
        # print(out2.size())
        # print(out3.size())
        # print(out4.size())
        # print(txt.size())

        out = [F.relu(conv(txt)).squeeze(3) for conv in self.convs]
        out = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in out]
        out = torch.cat(out, 1)  # 从第1维进行cat，将同一个样本的在不同一维卷积核的特征图进行cat。
        out = self.dropout1(out)
        out = F.relu(self.fc1(out))
        out = self.dropout2(out)
        logit = self.fc2(out)

        return logit
