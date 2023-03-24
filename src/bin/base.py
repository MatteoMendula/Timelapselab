import os
from ultralytics import YOLO
from src.utils import get_framework_extension


class BaseBin:
    def __init__(self,
                 version: str = '8',
                 size: str = 's',
                 dataset_name: str = 'timelapse'):
        self.version = version
        self.size = size
        self.dataset_name = dataset_name
        self.model = None
        self.yolo_weights_path = None

    def load_best_model(self, inference_mode: str = 'torch'):
        models_dir = os.path.join('yolo_models')
        if os.path.exists(os.path.join(models_dir, 'fine_tuned')):
            yolo_dir = os.path.join(models_dir, 'fine_tuned', self.dataset_name,
                                    'yolov{}{}'.format(self.version, self.size))
            try:
                self.model = YOLO(os.path.join(yolo_dir,
                                               'best.{}'.format(get_framework_extension(inference_mode))))
            except FileNotFoundError:
                self.model = YOLO(os.path.join(yolo_dir, 'best.pt'))
                self.model.export(format=get_framework_extension(inference_mode))
                self.model = YOLO(os.path.join(yolo_dir,
                                               'best.{}'.format(get_framework_extension(inference_mode))))
            self.yolo_weights_path = os.path.join(yolo_dir, 'best.{}'.format(get_framework_extension(inference_mode)))
        else:
            raise ValueError('Best model not available for version \'{}\' and size \'{}\''.format(self.version,
                                                                                                  self.size))
