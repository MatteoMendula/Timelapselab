import os
from ultralytics import YOLO


class BaseBin:
    def __init__(self,
                 version: str = '8',
                 size: str = 's'):
        self.version = version
        self.size = size
        self.model = None

    def load_best_model(self, dataset_name: str = 'timelapse'):
        models_dir = os.path.join('yolo_models')
        if os.path.exists(os.path.join(models_dir, 'fine_tuned')):
            self.model = YOLO(os.path.join(models_dir,
                                           'fine_tuned',
                                           dataset_name,
                                           'yolov{}{}'.format(self.version, self.size),
                                           'best.pt'))
        else:
            raise ValueError('Best model not available for version \'{}\' and size \'{}\''.format(self.version,
                                                                                                  self.size))
