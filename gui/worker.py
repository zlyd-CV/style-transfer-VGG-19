import sys
import os
from PyQt6.QtCore import QThread, pyqtSignal

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TrainingWorker(QThread):
    progress_signal = pyqtSignal(int)
    image_signal = pyqtSignal(str)
    finished_signal = pyqtSignal()
    error_signal = pyqtSignal(str)

    def __init__(self, content_path, style_path, output_dir, epochs=100):
        super().__init__()
        self.content_path = content_path
        self.style_path = style_path
        self.output_dir = output_dir
        self.epochs = epochs
        self._is_running = True

    def run(self):
        try:
            from code.train import train

            def callback(epoch, image_path):
                if not self._is_running:
                    raise InterruptedError("训练已停止")
                self.progress_signal.emit(epoch)
                if image_path and os.path.exists(image_path):
                    self.image_signal.emit(image_path)

            train(self.content_path, self.style_path,
                  self.epochs, callback, self.output_dir)

            if self._is_running:
                self.finished_signal.emit()

        except InterruptedError:
            pass
        except Exception as e:
            import traceback
            self.error_signal.emit(f"{str(e)}\n\n{traceback.format_exc()}")

    def stop(self):
        self._is_running = False
