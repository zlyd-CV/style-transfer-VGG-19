import torch
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class config:
    # 图像路径
    content_image_path = os.path.join(BASE_DIR, "data", "content_images")
    style_image_path = os.path.join(BASE_DIR, "data", "style_images")
    composite_image_path = os.path.join(BASE_DIR, "data", "composite_images")

    # 设备和训练参数
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_epochs = 100

    # 损失权重
    content_weight = 1e5
    style_weight = 1e10
    tv_weight = 10
