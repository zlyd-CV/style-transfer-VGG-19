# (Neural)-Style-Transfer-VGG-19

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)


## 一.项目介绍
+ 本项目是基于PyTorch实现的神经风格迁移项目，使用与训练好的VGG19网络提取特征，使用PyQt6开发的图形化界面操作。可支持自定义上传的图像与自定义的艺术风格图像结合，生成独特的艺术作品。


## 二. 内容介绍
+ 安装需要的包
```bash
pip install -r requirements.txt
```
+ 项目结构

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
├── main_gui.py           # GUI程序入口(运行该程序即可启动交互UI选择内容与样式进行训练)
├── requirements.txt      # 项目依赖
├── .gitignore           # Git忽略规则
├── LICENSE              # MIT开源协议
└── README.md            # 项目说明
```


## 三.部分关键参数
| 参数 | 增大效果 | 减小效果 |
|------|---------|---------|
| `content_weight` | 更接近原始内容图 | 风格更突出 |
| `style_weight` | 风格化更强烈 | 保留更多原图细节 |
| `tv_weight` | 图像更平滑 | 保留更多纹理细节 |

---


## 四.运行展示
![效果展示](https://github.com/zlyd-CV/Photos-Are-Used-To-Others-Repository/blob/473693ce60b8bfbc975dc4a4e59f9f47f25c7a21/style-transfer-VGG-19/50b8bc2b9aeef5af0760ff60bae36c0b.png)
![效果展示](https://github.com/zlyd-CV/Photos-Are-Used-To-Others-Repository/blob/473693ce60b8bfbc975dc4a4e59f9f47f25c7a21/style-transfer-VGG-19/659de723a38e42570f15666b6b7c9386.png)


## 五.部分资源下载地址
+ pytorch官网下载带cuda的pytorch：https://pytorch.org
+ Anaconda官网下载地址：https://anaconda.org/anaconda/conda

⭐ 如果这个项目对你有帮助，请给个Star支持一下！
