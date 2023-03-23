from typing import Union, List
from .base import BaseBin


class Converter(BaseBin):
    def __init__(self,
                 version: str = '8',
                 size: str = 's',
                 dataset_name: str = 'timelapse'):
        super().__init__(version=version, size=size, dataset_name=dataset_name)
        self.load_best_model()

    def convert_to(self, framework: Union[str, List[str]] = 'onnx'):
        if isinstance(framework, str):
            return self.model.export(format=framework)
        elif isinstance(framework, list):
            return {frame: self.model.export(format=frame) for frame in framework}
        else:
            raise ValueError('Cannot convert to framework having type \'{}\'!'.format(type(framework)))
