from typing import Union
import os
from ultralytics import YOLO


class Converter:
    def __init__(self,
                 version: str = '8',
                 size: str = 's'):
        self.version = version
        self.size = size
        self.model = None
        self.load_best_model()

    def load_best_model(self):
        models_dir = os.path.join('yolo_models')
        if os.path.exists(os.path.join(models_dir, 'fine_tuned')):
            self.model = YOLO(os.path.join(models_dir,
                                           'fine_tuned',
                                           'yolov{}{}'.format(self.version, self.size),
                                           'best.pt'))
        else:
            raise ValueError('Best model not available for version \'{}\' and size \'{}\''.format(self.version,
                                                                                                  self.size))

    def convert_to(self, framework: Union[str, list[str]] = 'onnx'):
        if isinstance(framework, str):
            return self.model.export(format=framework)
        elif isinstance(framework, list):
            return {frame: self.model.export(format=frame) for frame in framework}
        else:
            raise ValueError('Cannot convert to framework having type \'{}\'!'.format(type(framework)))
