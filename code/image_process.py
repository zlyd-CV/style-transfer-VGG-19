import os
import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as transforms
from .config import config

# ImageNet标准化参数
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

# 图像预处理流程
img_transforms = transforms.Compose([
    transforms.Resize(512),
    transforms.CenterCrop(512),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD)
])


def load_image(img_path):
    """加载并预处理图像"""
    img = Image.open(img_path).convert('RGB')
    return img_transforms(img).unsqueeze(0)


def save_image(img, filename="output.png", output_dir=None):
    """保存生成的图像"""
    img = img.squeeze(0).cpu()

    # 反标准化
    t_mean = torch.tensor(MEAN).view(3, 1, 1)
    t_std = torch.tensor(STD).view(3, 1, 1)
    img = (img * t_std + t_mean).clamp(0, 1)

    # 保存
    img = transforms.ToPILImage()(img)

    if output_dir is None:
        output_dir = config.composite_image_path

    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, filename)
    img.save(save_path)
    return save_path


class GeneratedImage(nn.Module):
    """可训练的生成图像，从内容图像初始化"""

    def __init__(self, content_path):
        super().__init__()
        self.params = nn.Parameter(load_image(
            content_path), requires_grad=True)

    def forward(self):
        return self.params
