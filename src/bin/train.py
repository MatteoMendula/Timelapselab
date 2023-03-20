from typing import Union
import os
import shutil
from ultralytics import YOLO


class Trainer:
    def __init__(self,
                 version: str = '8',
                 size: str = 's',
                 dataset: str = 'data/timelapse.yaml'):
        self.version = version
        self.size = size
        self.dataset = dataset
        self.model = None
        self.setup_model()

    def setup_model(self,
                    pretrained: bool = True):
        models_dir = os.path.join('yolo_models')
        if pretrained:
            model_name = 'yolov{}{}.pt'.format(self.version, self.size)
            self.model = YOLO(os.path.join(models_dir, 'pretrained', model_name))
        else:
            raise ValueError('Training from scratch is not available yet!')

    def run(self,
            batch_size: int = 64,
            img_size: int = 640,
            epochs: int = 10,
            device: Union[str, int, list[int]] = 0,
            val: bool = True) -> YOLO:
        self.model.add_callback("on_train_epoch_end", self.on_train_epoch_end)
        self.model.train(data=self.dataset,
                         batch=batch_size,
                         imgsz=img_size,
                         epochs=epochs,
                         device=device,
                         save=True,
                         save_period=1)
        if val:
            self.model.val()
        return self.model

    def get_model(self) -> YOLO:
        return self.model

    def on_train_epoch_end(self, model):
        models_dir = os.path.join('yolo_models')
        fine_tuned_dir = os.path.join(models_dir,
                                      'fine_tuned',
                                      os.path.basename(self.dataset).split('\\')[-1].split('.')[0],
                                      'yolov{}{}'.format(self.version, self.size))
        if not os.path.exists(fine_tuned_dir):
            os.makedirs(fine_tuned_dir)
            shutil.copyfile(os.path.join(self.model.save_dir, 'weights', 'best.pt'),
                            os.path.join(fine_tuned_dir, 'best.pt'))
            shutil.copyfile(os.path.join(self.model.save_dir, 'weights', 'last.pt'),
                            os.path.join(fine_tuned_dir, 'last.pt'))

