import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import VGG19_Weights
from .config import config

# 风格层和内容层索引（VGG19共37层）
style_layers = [0, 5, 10, 19, 28]
content_layers = [25]

# 延迟加载VGG模型，避免模块导入时占用过多资源
_vgg_cache = None


def get_vgg_model():
    """获取VGG19模型（单例模式）"""
    global _vgg_cache
    if _vgg_cache is None:
        _vgg_cache = models.vgg19(
            weights=VGG19_Weights.DEFAULT).features.to(config.device).eval()
    return _vgg_cache


class ContentLoss(nn.Module):
    def __init__(self, target, weight):
        super(ContentLoss, self).__init__()
        self.target = target.detach()  # 保存目标特征图，detach()表示不需要计算其梯度
        self.weight = weight
        # 使用均方误差作为损失函数,MSE= ∑(内容图特征值−生成图特征值)^2 / N
        self.criterion = nn.MSELoss()
        # 存储当前的损失值
        self.loss = None

    def forward(self, inputs):
        # 计算内容损失：新的生成图像特征(inputs)与原始内容图(target)的均方误差，并乘以权重
        self.loss = self.weight * self.criterion(inputs, self.target)

        # 克隆输入并返回，保持计算图的连续性
        outputs = inputs.clone()
        return outputs


class Gram(nn.Module):
    def __init__(self):
        super(Gram, self).__init__()

    def forward(self, inputs):
        # 获取特征图的维度信息
        batch_size, channels, width, height = inputs.size()

        # 将特征图重塑：从[batch, channels, height, width]变为[batch*channels, height*width]
        # 每一行代表一个特征通道在所有空间位置的激活值
        features = inputs.view(batch_size * channels, width * height)

        # 计算Gram矩阵：features与其转置相乘
        # 结果是一个[channels, channels]的矩阵，表示不同通道间的相关性
        # 然后除以总元素数进行归一化
        return torch.mm(features, features.t()) / (batch_size * channels * width * height)


class StyleLoss(nn.Module):
    def __init__(self, target, weight):
        super(StyleLoss, self).__init__()

        # 保存目标Gram矩阵，detach()表示不需要计算其梯度
        self.target = target.detach()

        # 风格损失的权重
        self.weight = weight

        # Gram矩阵计算器
        self.gram = Gram()

        # 使用均方误差作为损失函数
        self.criterion = nn.MSELoss()

    def forward(self, inputs):
        # 计算生成图像特征的Gram矩阵
        gram_features = self.gram(inputs)

        # 计算生成图像的Gram矩阵与目标Gram矩阵的均方误差，并乘以权重
        self.loss = self.weight * self.criterion(gram_features, self.target)

        # 克隆输入并返回，保持计算图的连续性
        outputs = inputs.clone()
        return outputs


class TotalVariationDenoisingLoss(nn.Module):
    def __init__(self, weight):
        super(TotalVariationDenoisingLoss, self).__init__()

        # 总变差损失的权重
        self.weight = weight

        # 存储当前的损失值
        self.loss = None

    def forward(self, inputs):
        vertical_diff = (inputs[:, :, 1:, :] -
                         inputs[:, :, :-1, :]).abs().mean()

        horizontal_diff = (inputs[:, :, :, 1:] -
                           inputs[:, :, :, :-1]).abs().mean()

        # 总损失 = 权重 × 0.5 × (垂直差异 + 水平差异)
        self.loss = self.weight * 0.5 * (vertical_diff + horizontal_diff)

        # 克隆输入并返回，保持计算图的连续性
        outputs = inputs.clone()
        return outputs


class StyleTransferModel(nn.Module):
    def __init__(self, style_img, content_img, base_model=None):
        """
        参数：
            :param style_img: 风格图像（提供艺术风格的图像）
            :param content_img: 内容图像（提供内容结构的图像）
            :param base_model: 基础特征提取网络，默认使用VGG19
        """
        super().__init__()
        self.style_img = style_img
        self.content_img = content_img
        self.base_model = base_model if base_model is not None else get_vgg_model()
        self.layers = nn.Sequential()

    def generate_layers(self):
        # 初始化三个列表，用于存储各类损失层的引用
        content_loss_list, style_loss_list, total_variation_denoising_loss_list = [], [], []

        # 初始化各类层的索引计数器，用于生成唯一的层名称
        conv2d_index, maxpool2d_index, batchnorm2d_index, relu_index = 1, 1, 1, 1

        # 添加总变差去噪损失层
        tv_loss = TotalVariationDenoisingLoss(config.tv_weight)
        self.layers.add_module("TV_Loss", tv_loss)
        total_variation_denoising_loss_list.append(tv_loss)

        # 遍历VGG19的所有层
        for idx, layer in enumerate(self.base_model):
            # ========== 处理卷积层 ==========
            if isinstance(layer, nn.Conv2d):
                # 添加卷积层到模型
                name = "Conv2d_{}".format(conv2d_index)
                self.layers.add_module(name, layer)

                if idx in content_layers:
                    target = self.layers(self.content_img)
                    content_loss = ContentLoss(target, config.content_weight)
                    self.layers.add_module(
                        f"Content_Loss_{conv2d_index}", content_loss)
                    content_loss_list.append(content_loss)

                elif idx in style_layers:
                    target = Gram()(self.layers(self.style_img))
                    style_loss = StyleLoss(target, config.style_weight)
                    self.layers.add_module(
                        f"Style_Loss_{conv2d_index}", style_loss)
                    style_loss_list.append(style_loss)

                # 卷积层计数器加1
                conv2d_index += 1

            elif isinstance(layer, nn.MaxPool2d):
                # 添加池化层到模型（用于下采样，减小特征图尺寸）
                name = "MaxPool2d_{}".format(maxpool2d_index)
                self.layers.add_module(name, layer)
                maxpool2d_index += 1

            elif isinstance(layer, nn.ReLU):
                # 添加激活层到模型（引入非线性）
                name = "Relu_{}".format(relu_index)
                self.layers.add_module(name, layer)
                relu_index += 1

            else:
                # 添加批归一化层到模型（用于加速训练和稳定性）
                name = "BatchNorm2d_{}".format(batchnorm2d_index)
                self.layers.add_module(name, layer)
                batchnorm2d_index += 1

        # 返回所有损失层的引用列表
        return content_loss_list, style_loss_list, total_variation_denoising_loss_list

    def forward(self, inputs):
        outputs = self.layers(inputs)
        return outputs
