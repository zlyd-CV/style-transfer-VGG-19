import os
import shutil
from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QPushButton, QLabel, QFileDialog, QProgressBar, QMessageBox, QScrollArea
)
from PyQt6.QtGui import QPixmap, QFont
from PyQt6.QtCore import Qt
from gui.worker import TrainingWorker


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("🎨 风格迁移")
        self.setGeometry(50, 50, 1400, 900)

        # 路径配置
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.data_dir = os.path.join(base_dir, 'data')
        self.temp_dir = os.path.join(self.data_dir, 'temp_images')
        self.composite_dir = os.path.join(self.data_dir, 'composite_images')

        for d in ['content_images', 'style_images', 'temp_images', 'composite_images']:
            os.makedirs(os.path.join(self.data_dir, d), exist_ok=True)

        self.content_path = None
        self.style_path = None
        self.worker = None
        self.image_widgets = []

        self.init_ui()

    def init_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setSpacing(10)
        layout.setContentsMargins(15, 15, 15, 15)

        # 标题
        title = QLabel("🎨 神经网络风格迁移")
        title.setFont(QFont('Arial', 20, QFont.Weight.Bold))
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setFixedHeight(40)
        layout.addWidget(title)

        # 顶部控制区
        top_layout = QHBoxLayout()
        top_layout.setSpacing(10)

        # 创建图像选择区（内容+风格）
        for label_text, btn_text, btn_color, callback in [
            ("内容图像", "📂 选择内容", "#007bff", self.select_content_image),
            ("风格图像", "🎨 选择风格", "#28a745", self.select_style_image)
        ]:
            box = QVBoxLayout()
            lbl = QLabel(label_text)
            lbl.setFixedSize(250, 180)
            lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            lbl.setStyleSheet(
                "border: 2px dashed #999; border-radius: 5px; background: #f5f5f5;")

            btn = QPushButton(btn_text)
            btn.setFixedSize(120, 35)
            btn.setStyleSheet(
                f"background: {btn_color}; color: white; border-radius: 5px; font-weight: bold;")
            btn.clicked.connect(callback)

            box.addWidget(lbl)
            box.addWidget(btn, alignment=Qt.AlignmentFlag.AlignCenter)
            top_layout.addLayout(box)

            if label_text == "内容图像":
                self.content_label = lbl
                self.content_btn = btn
            else:
                self.style_label = lbl
                self.style_btn = btn

        # 控制按钮和进度条
        control_box = QVBoxLayout()
        self.start_btn = QPushButton("▶ 开始训练")
        self.start_btn.setFixedHeight(45)
        self.start_btn.setEnabled(False)
        self.start_btn.setStyleSheet("""
            QPushButton { background: #6c757d; color: white; border-radius: 8px; font-size: 16px; font-weight: bold; }
            QPushButton:enabled { background: #28a745; }
            QPushButton:enabled:hover { background: #218838; }
        """)
        self.start_btn.clicked.connect(self.start_training)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setFixedHeight(25)
        self.progress_bar.setFormat("进度: %p% (Epoch %v/100)")
        self.progress_bar.setStyleSheet("""
            QProgressBar { border: 2px solid #ddd; border-radius: 5px; text-align: center; font-weight: bold; }
            QProgressBar::chunk { background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #28a745, stop:1 #20c997); }
        """)

        control_box.addWidget(self.start_btn)
        control_box.addWidget(self.progress_bar)
        control_box.addStretch()
        top_layout.addLayout(control_box)
        top_layout.addStretch()
        layout.addLayout(top_layout)

        # 结果展示区
        result_title = QLabel("✨ 生成结果 (每10轮保存一次)")
        result_title.setFont(QFont('Arial', 14, QFont.Weight.Bold))
        result_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        result_title.setFixedHeight(30)
        layout.addWidget(result_title)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet(
            "QScrollArea { border: 2px solid #ddd; border-radius: 5px; }")

        scroll_content = QWidget()
        grid = QGridLayout(scroll_content)
        grid.setSpacing(15)

        # 创建10个图像位置
        for i in range(10):
            widget = QWidget()
            box = QVBoxLayout(widget)
            box.setSpacing(5)
            box.setContentsMargins(5, 5, 5, 5)

            label = QLabel(f"Epoch {i*10}")
            label.setFixedSize(250, 180)
            label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            label.setStyleSheet(
                "border: 1px solid #ddd; background: white; color: #999;")

            save_btn = QPushButton("💾 保存")
            save_btn.setFixedHeight(30)
            save_btn.setEnabled(False)
            save_btn.setStyleSheet("""
                QPushButton { background: #6c757d; color: white; border-radius: 3px; font-weight: bold; }
                QPushButton:enabled { background: #17a2b8; }
                QPushButton:enabled:hover { background: #138496; }
            """)
            save_btn.clicked.connect(
                lambda checked, idx=i: self.save_single_image(idx))

            box.addWidget(label)
            box.addWidget(save_btn)
            grid.addWidget(widget, i // 5, i % 5)
            self.image_widgets.append(
                {'label': label, 'button': save_btn, 'path': None})

        scroll.setWidget(scroll_content)
        layout.addWidget(scroll, stretch=1)

    def select_content_image(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "选择内容图像", "", "Images (*.png *.jpg *.jpeg)")
        if path:
            dest = os.path.join(
                self.data_dir, 'content_images', os.path.basename(path))
            shutil.copy(path, dest)
            self.content_path = dest
            pixmap = QPixmap(dest).scaled(
                400, 300, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
            self.content_label.setPixmap(pixmap)
            self.check_ready()

    def select_style_image(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "选择风格图像", "", "Images (*.png *.jpg *.jpeg)")
        if path:
            dest = os.path.join(
                self.data_dir, 'style_images', os.path.basename(path))
            shutil.copy(path, dest)
            self.style_path = dest
            pixmap = QPixmap(dest).scaled(
                400, 300, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
            self.style_label.setPixmap(pixmap)
            self.check_ready()

    def check_ready(self):
        if self.content_path and self.style_path:
            self.start_btn.setEnabled(True)

    def start_training(self):
        """开始训练"""
        self._toggle_buttons(False)
        self.progress_bar.setValue(0)

        # 清空旧结果
        for w in self.image_widgets:
            w['label'].clear()
            w['label'].setText("处理中...")
            w['button'].setEnabled(False)
            w['path'] = None

        # 清理临时目录
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        os.makedirs(self.temp_dir, exist_ok=True)

        # 启动训练线程
        self.worker = TrainingWorker(
            self.content_path, self.style_path, self.temp_dir)
        self.worker.progress_signal.connect(self.update_progress)
        self.worker.image_signal.connect(self.update_image)
        self.worker.finished_signal.connect(self.training_finished)
        self.worker.error_signal.connect(self.training_error)
        self.worker.start()

    def _toggle_buttons(self, enabled):
        """切换按钮状态"""
        self.start_btn.setEnabled(enabled)
        self.content_btn.setEnabled(enabled)
        self.style_btn.setEnabled(enabled)
        # 确定是第几张图像（epoch / 10）
        # 文件名格式：epoch_0.png, epoch_10.png, ..., epoch_90.png
        basename = os.path.basename(path)
        epoch_str = basename.replace('epoch_', '').replace(
            '.png', '').replace('.jpg', '').replace('.jpeg', '')
        epoch_num = int(epoch_str)

    def update_progress(self, epoch):
        self.progress_bar.setValue(epoch + 1)

    def update_image(self, path):
        """更新图像显示"""
        try:
            basename = os.path.basename(path)
            epoch_num = int(basename.replace('epoch_', '').split('.')[0])
            idx = epoch_num // 10

            if 0 <= idx < 10 and os.path.exists(path):
                pixmap = QPixmap(path)
                if not pixmap.isNull():
                    self.image_widgets[idx]['label'].setPixmap(
                        pixmap.scaled(250, 180, Qt.AspectRatioMode.KeepAspectRatio,
                                      Qt.TransformationMode.SmoothTransformation))
                    self.image_widgets[idx]['button'].setEnabled(True)
                    self.image_widgets[idx]['path'] = path
        except Exception as e:
            print(f"更新图像出错: {e}")

    def training_finished(self):
        """训练完成后的处理"""
        self.start_btn.setEnabled(True)
        self.content_btn.setEnabled(True)
        self.style_btn.setEnabled(True)
        QMessageBox.information(self, "完成", "训练已完成！所有图像已生成。")

    def training_error(self, error):
        """训练出错时的处理"""
        self.start_btn.setEnabled(True)
        self.content_btn.setEnabled(True)
        self.style_btn.setEnabled(True)
        QMessageBox.critical(self, "错误", f"训练出错：\n{error}")

    def closeEvent(self, event):
        """窗口关闭时的清理工作"""

    def save_single_image(self, idx):
        if self.image_widgets[idx]['path']:
            source = self.image_widgets[idx]['path']
            default = os.path.join(
                self.composite_dir, f"result_epoch_{idx*10}.png")
            path, _ = QFileDialog.getSaveFileName(
                self, "保存图像", default, "PNG (*.png);;JPEG (*.jpg)")
            if path:
                shutil.copy(source, path)
                QMessageBox.information(self, "成功", f"图像已保存！\n{path}")

    def training_finished(self):
        self._toggle_buttons(True)
        QMessageBox.information(self, "完成", "训练已完成！")

    def training_error(self, error):
        self._toggle_buttons(True)
        QMessageBox.critical(self, "错误", f"训练出错：\n{error}")

    def closeEvent(self, event):
        if self.worker and self.worker.isRunning():
            self.worker.stop()
            self.worker.wait(5000)
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        event.accept()
