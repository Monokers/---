import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        """
        初始化残差块模块
        in_channels: 输入通道数
        out_channels: 输出通道数
        stride: 卷积步长，默认为 1
        """
        super(ResidualBlock, self).__init__()
        # 第一个卷积层
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        # 第二个卷积层
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # 快捷连接，如果输入输出通道数不同或步长不为 1，则需要进行调整
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        """
        前向传播函数
        x: 输入特征图
        return: 输出特征图
        """
        residual = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        # 加上快捷连接的结果
        x += self.shortcut(residual)
        x = self.relu(x)
        return x


class ImageClassificationModel(nn.Module):
    def __init__(self, num_classes):
        """
        初始化图像分类模型
        :param num_classes: 分类的类别数
        """
        super(ImageClassificationModel, self).__init__()
        # 初始卷积层
        self.initial_conv = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # 使用 ResidualBlock 模块
        self.res_block1 = ResidualBlock(64, 64, stride=1)
        self.res_block2 = ResidualBlock(64, 128, stride=2)
        self.res_block3 = ResidualBlock(128, 256, stride=2)

        # 全局平均池化层
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        # Dropout 层
        #self.dropout = nn.Dropout(0.5)
        # 全连接层
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        """
        前向传播函数
        :param x: 输入图像数据
        :return: 输出预测结果
        """
        x = self.initial_conv(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.max_pool(x)

        x = self.res_block1(x)
        x = self.res_block2(x)
        x = self.res_block3(x)

        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)
        #x = self.dropout(x)
        x = self.fc(x)
        return x

