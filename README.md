# 🎨 神经网络风格迁移 Neural Style Transfer

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

基于PyTorch实现的神经风格迁移项目，使用VGG19网络提取特征，支持图形化界面操作。将任意内容图像与艺术风格图像结合，生成独特的艺术作品。

## ✨ 特性

- 🖼️ **图形化界面** - 基于PyQt6的现代化UI，无需命令行操作
- ⚡ **GPU加速** - 自动检测CUDA，支持GPU加速训练
- 📊 **实时预览** - 训练过程中每10轮自动更新预览
- 💾 **批量保存** - 10个检查点图像可独立保存
- 🎯 **参数可调** - 灵活调整内容/风格权重

## 📦 安装

### 1. 克隆项目

```bash
git clone https://github.com/yourusername/neural-style-transfer.git
cd neural-style-transfer
```

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

或手动安装：

```bash
pip install torch torchvision PyQt6 Pillow tqdm
```

## 🚀 快速开始

运行图形界面：

```bash
python main_gui.py
```

## 📁 项目结构

```
neural-style-transfer/
├── code/                  # 核心算法实现
│   ├── config.py         # 全局配置参数
│   ├── model.py          # VGG19模型和损失函数
│   ├── train.py          # 训练逻辑
│   └── image_process.py  # 图像加载/保存/预处理
├── gui/                   # 图形界面
│   ├── app.py            # 主窗口界面
│   └── worker.py         # 后台训练线程
├── data/                  # 数据文件夹
│   ├── content_images/   # 内容图像目录
│   ├── style_images/     # 风格图像目录
│   ├── composite_images/ # 最终输出目录
│   └── temp_images/      # 临时文件（自动清理）
├── main_gui.py           # GUI程序入口
├── requirements.txt      # 项目依赖
├── .gitignore           # Git忽略规则
├── LICENSE              # MIT开源协议
└── README.md            # 项目说明
```

## 🎯 使用说明

### 图形界面操作

1. **选择内容图像** - 点击"📂 选择内容"上传你的照片
2. **选择风格图像** - 点击"🎨 选择风格"上传艺术作品
3. **开始训练** - 点击"▶ 开始训练"按钮（默认100轮迭代）
4. **查看结果** - 训练过程中每10轮自动显示结果（共10张预览图）
5. **保存图像** - 点击每张预览图下方的"💾 保存"按钮导出

### 训练时间参考

- **GPU (CUDA)**: 约2-3分钟
- **CPU**: 约15-20分钟

## ⚙️ 参数调整

在 `code/config.py` 中自定义参数：

```python
class config:
    # 损失权重
    content_weight = 1e5    # 内容保留强度（越大越接近原图）
    style_weight = 1e10     # 风格应用强度（越大风格越明显）
    tv_weight = 10          # 总变差损失（控制图像平滑度）
    
    # 训练参数
    num_epochs = 100        # 训练轮数
    device = "cuda"         # 或 "cpu"
```

### 参数效果说明

| 参数 | 增大效果 | 减小效果 |
|------|---------|---------|
| `content_weight` | 更接近原始内容图 | 风格更突出 |
| `style_weight` | 风格化更强烈 | 保留更多原图细节 |
| `tv_weight` | 图像更平滑 | 保留更多纹理细节 |

## 🧠 技术原理

### 核心算法

- **特征提取**: VGG19预训练模型（ImageNet）
- **内容损失**: 高层特征的MSE损失
- **风格损失**: Gram矩阵的MSE损失
- **优化器**: L-BFGS二阶优化
- **总变差正则**: 减少噪点，提升平滑度

### 损失函数

```
Total Loss = α × Content Loss + β × Style Loss + γ × TV Loss
```

## 💡 使用建议

1. **图像尺寸** - 默认处理为512x512，过大图像会自动缩放
2. **风格图选择** - 推荐使用高对比度、纹理丰富的艺术作品
3. **GPU推荐** - 使用CUDA可大幅加速训练（约10倍速度提升）
4. **多次尝试** - 不同权重参数会产生截然不同的效果

## 🖼️ 效果展示

*（建议添加一些示例图片展示效果）*

## 📝 开发计划

- [ ] 支持自定义训练轮数
- [ ] 添加预设风格模板
- [ ] 支持视频风格迁移
- [ ] 实时风格迁移（FastNeuralStyle）
- [ ] 命令行模式（批量处理）

## 🤝 贡献

欢迎提交Issue和Pull Request！

1. Fork本项目
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启Pull Request

## 📄 开源协议

本项目采用 [MIT License](LICENSE) 开源协议。

## 🙏 致谢

- [PyTorch](https://pytorch.org/) - 深度学习框架
- [VGG19](https://arxiv.org/abs/1409.1556) - 预训练模型
- [Neural Style Transfer论文](https://arxiv.org/abs/1508.06576) - Gatys et al.

## 📧 联系方式

如有问题或建议，请提交Issue或联系开发者。

---

⭐ 如果这个项目对你有帮助，请给个Star支持一下！
