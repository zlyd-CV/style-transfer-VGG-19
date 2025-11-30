import os
import torch
import torch.optim as optim
from tqdm import tqdm
from .model import StyleTransferModel
from .image_process import load_image, save_image, GeneratedImage
from .config import config


def train(content_path=None, style_path=None, epochs=None, callback=None, output_dir=None):
    # 使用参数或默认配置
    if content_path is None:
        content_path = os.path.join(config.content_image_path, "content.jpg")
    if style_path is None:
        style_path = os.path.join(config.style_image_path, "style.jpg")
    if epochs is None:
        epochs = config.num_epochs

    # 加载图像
    style_img = load_image(style_path).to(config.device)
    content_img = load_image(content_path).to(config.device)

    # 初始化模型
    input_img = GeneratedImage(content_path).to(config.device)
    model = StyleTransferModel(style_img, content_img).to(config.device)

    content_loss_list, style_loss_list, tv_loss_list = model.generate_layers()
    optimizer = optim.LBFGS([input_img.params])

    # 训练循环
    pbar = tqdm(range(epochs), desc="训练进度", disable=callback is not None)

    for epoch in pbar:
        def closure():
            optimizer.zero_grad()
            model(input_img())

            # 计算总损失
            style_loss = sum(sl.loss for sl in style_loss_list)
            content_loss = sum(cl.loss for cl in content_loss_list)
            tv_loss = tv_loss_list[0].loss
            total_loss = content_loss + style_loss + tv_loss

            total_loss.backward()
            return total_loss

        optimizer.step(closure)

        # 每10个epoch保存并回调
        if epoch % 10 == 0:
            try:
                filename = f"epoch_{epoch}.png"
                # 使用clone().detach()确保完全断开梯度图
                saved_path = save_image(
                    input_img.params.clone().detach(), filename, output_dir)

                if callback:
                    callback(epoch, saved_path)
                else:
                    pbar.set_postfix({
                        'C': f'{sum(cl.loss for cl in content_loss_list).item():.0f}',
                        'S': f'{sum(sl.loss for sl in style_loss_list).item():.0f}',
                        'TV': f'{tv_loss_list[0].loss.item():.2f}'
                    })
            except Exception as e:
                print(f"保存图像时出错: {e}")
                import traceback
                traceback.print_exc()


# ========== 主程序入口 ==========
if __name__ == '__main__':
    # 当直接运行此脚本时，开始训练
    train()
