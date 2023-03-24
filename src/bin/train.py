from typing import Union, List
import os
import shutil
from ultralytics import YOLO
from .base import BaseBin


class Trainer(BaseBin):
    def __init__(self,
                 version: str = '8',
                 size: str = 's',
                 dataset_path: str = 'data/timelapse.yaml'):
        dataset_name = os.path.basename(dataset_path).split('\\')[-1].split('.')[0]
        super().__init__(version=version, size=size, dataset_name=dataset_name)
        self.dataset_path = dataset_path
        self.setup_model()
        self.img_size = None

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
            img_size: Union[int, List[int]] = 640,
            epochs: int = 10,
            device: Union[str, int, List[int]] = 0,
            val: bool = True) -> YOLO:
        self.img_size = img_size
        self.model.add_callback("on_fit_epoch_end", self.on_fit_epoch_end)
        print('\n\nimg_size: {}\n\n'.format(img_size))
        self.model.train(data=self.dataset_path,
                         batch=batch_size,
                         imgsz=img_size,
                         epochs=epochs,
                         device=device,
                         save=True,
                         save_period=1,
                         workers=1)
        if val:
            self.model.val()
        return self.model

    def get_model(self) -> YOLO:
        return self.model

    def on_fit_epoch_end(self, model):
        models_dir = os.path.join('yolo_models')
        fine_tuned_dir = os.path.join(models_dir,
                                      'fine_tuned',
                                      self.dataset_name,
                                      'yolov{}{}'.format(self.version, self.size))
        if not os.path.exists(fine_tuned_dir):
            os.makedirs(fine_tuned_dir)
        print('Copying best and last models to yolo_models folder...')
        shutil.copyfile(os.path.join(self.model.trainer.save_dir, 'weights', 'best.pt'),
                        os.path.join(fine_tuned_dir, 'best.pt'))
        shutil.copyfile(os.path.join(self.model.trainer.save_dir, 'weights', 'last.pt'),
                        os.path.join(fine_tuned_dir, 'last.pt'))
        print('Best and last models were copied correctly to yolo_models folder')

    def export(self, framework: Union[str, List[str]] = 'onnx'):
        self.load_best_model()
        if isinstance(framework, str):
            return self.model.export(format=framework,
                                     imgsz=self.img_size)
        elif isinstance(framework, list):
            return {frame: self.model.export(format=frame,
                                             imgsz=self.img_size) for frame in framework}
        else:
            raise ValueError('Cannot convert to framework having type \'{}\'!'.format(type(framework)))
